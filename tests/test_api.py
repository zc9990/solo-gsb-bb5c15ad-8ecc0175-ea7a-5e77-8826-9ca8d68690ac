"""端到端 API 测试：每个用例使用独立的临时 SQLite 数据库。"""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import database, service
from app.main import app

# 纯西里尔字母 "арр"（а U+0430, р U+0440），与拉丁 "app" 同形
CYRILLIC_APP = "арр"
# 拉丁 "ad" + 西里尔 "м"（U+043C）+ 拉丁 "in" —— 混脚本
MIXED_ADMIN = "adмin"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(db_path))
    database.init_db(str(db_path))
    with TestClient(app) as test_client:
        yield test_client


def register(client, name, namespace="chat"):
    return client.post(f"/namespaces/{namespace}/names", json={"name": name})


def test_register_returns_201_and_stores_original_and_normalized(client):
    resp = register(client, "Ａｌｉｃｅ")  # 全角字母
    assert resp.status_code == 201
    body = resp.json()
    assert body["original_name"] == "Ａｌｉｃｅ"  # 原名落库
    assert body["normalized_name"] == "alice"  # NFKC + casefold
    assert body["namespace"] == "chat"
    assert body["id"] >= 1


def test_normalized_duplicate_returns_409_with_conflict_record(client):
    assert register(client, "Alice").status_code == 201

    resp = register(client, "ALICE")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "name_conflict"
    assert detail["normalized_name"] == "alice"
    assert detail["conflicts"][0]["original_name"] == "Alice"
    assert detail["conflicts"][0]["reason"] == "normalized_match"

    # NFKC 兼容字符（全角）折叠后同样冲突
    resp = register(client, "Ａｌｉｃｅ")
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflicts"][0]["reason"] == "normalized_match"


def test_confusable_name_returns_409(client):
    assert register(client, "app").status_code == 201

    resp = register(client, CYRILLIC_APP)  # 纯西里尔，单脚本不触发 422
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["conflicts"][0]["reason"] == "confusable"
    assert detail["conflicts"][0]["original_name"] == "app"

    # 反向同样成立：先注册纯西里尔，再注册拉丁
    assert register(client, CYRILLIC_APP, namespace="other").status_code == 201
    resp = register(client, "app", namespace="other")
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflicts"][0]["reason"] == "confusable"


def test_mixed_latin_cyrillic_returns_422_with_code_points(client):
    resp = register(client, MIXED_ADMIN)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "mixed_script_violation"
    assert detail["restricted_scripts_used"] == ["Cyrillic", "Latin"]

    by_script = {d["script"]: d for d in detail["script_details"]}
    assert by_script["Cyrillic"]["characters"] == ["м"]
    assert by_script["Cyrillic"]["code_points"] == ["U+043C"]
    assert set(by_script["Latin"]["characters"]) == {"a", "d", "i", "n"}
    assert "U+0061" in by_script["Latin"]["code_points"]

    # 422 不落库
    assert client.get("/namespaces/chat/names").json() == []


def test_mixed_latin_greek_returns_422(client):
    resp = register(client, "Αlpha")  # 希腊大写 Alpha + 拉丁
    assert resp.status_code == 422
    assert resp.json()["detail"]["restricted_scripts_used"] == ["Greek", "Latin"]


def test_single_restricted_script_is_allowed(client):
    assert register(client, "αρρ").status_code == 201  # 纯西里尔
    assert register(client, "ΑΒΓ").status_code == 201  # 纯希腊
    assert register(client, "hello-123_世界").status_code == 201  # Latin+Han 不拦截


def test_check_endpoint_analyzes_without_persisting(client):
    assert register(client, "app").status_code == 201

    resp = client.post("/namespaces/chat/check", json={"name": CYRILLIC_APP})
    assert resp.status_code == 200
    body = resp.json()
    assert body["normalized_name"] == CYRILLIC_APP
    assert body["mixed_script_violation"] is False
    assert body["scripts"] == ["Cyrillic"]
    assert len(body["conflicts"]) == 1
    assert body["conflicts"][0]["reason"] == "confusable"

    # 检查接口不落库：命名空间内仍只有 1 条记录
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["app"]


def test_check_endpoint_reports_mixed_scripts_without_persisting(client):
    resp = client.post("/namespaces/chat/check", json={"name": MIXED_ADMIN})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mixed_script_violation"] is True
    assert body["restricted_scripts_used"] == ["Cyrillic", "Latin"]
    assert client.get("/namespaces/chat/names").json() == []


def test_namespaces_are_isolated(client):
    assert register(client, "root", namespace="a").status_code == 201
    assert register(client, "root", namespace="b").status_code == 201
    assert len(client.get("/namespaces/a/names").json()) == 1
    assert len(client.get("/namespaces/b/names").json()) == 1


def test_blank_name_rejected(client):
    assert register(client, "   ").status_code == 422
    assert register(client, "").status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# ---------- 批量注册 POST /namespaces/{namespace}/names/batch ----------


def register_batch(client, names, namespace="chat"):
    return client.post(f"/namespaces/{namespace}/names/batch", json={"names": names})


def test_batch_register_returns_201_and_persists_all_in_order(client):
    resp = register_batch(client, ["Alice", "Ｂｏｂ", "世界-123"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["namespace"] == "chat"
    assert [item["index"] for item in body["items"]] == [0, 1, 2]
    assert [item["original_name"] for item in body["items"]] == ["Alice", "Ｂｏｂ", "世界-123"]
    assert [item["normalized_name"] for item in body["items"]] == ["alice", "bob", "世界-123"]
    assert all(item["id"] >= 1 and item["created_at"] for item in body["items"])

    # 全部落库，且可通过列表接口查到
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["Alice", "Ｂｏｂ", "世界-123"]


def test_batch_max_size_50_accepted(client):
    resp = register_batch(client, [f"user{i}" for i in range(50)])
    assert resp.status_code == 201
    assert len(resp.json()["items"]) == 50


def test_batch_size_limits_rejected(client):
    assert register_batch(client, []).status_code == 422  # 至少 1 个
    assert register_batch(client, [f"user{i}" for i in range(51)]).status_code == 422  # 至多 50 个
    assert register_batch(client, ["ok", "   "]).status_code == 422  # 空白名称
    assert client.get("/namespaces/chat/names").json() == []


def test_batch_in_batch_normalized_duplicate_returns_422_and_marks_both(client):
    resp = register_batch(client, ["Alice", "bob", "ALICE"])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "batch_validation_failed"

    items = {item["index"]: item for item in detail["items"]}
    assert set(items) == {0, 2}  # bob 无问题，不出现在明细中
    assert items[0]["batch_conflicts"][0]["index"] == 2
    assert items[0]["batch_conflicts"][0]["original_name"] == "ALICE"
    assert items[0]["batch_conflicts"][0]["reason"] == "normalized_match"
    assert items[2]["batch_conflicts"][0]["index"] == 0
    assert items[2]["batch_conflicts"][0]["original_name"] == "Alice"

    # 422 整批不落库
    assert client.get("/namespaces/chat/names").json() == []


def test_batch_in_batch_confusable_returns_422_and_marks_both(client):
    resp = register_batch(client, ["app", CYRILLIC_APP, "ok-name"])
    assert resp.status_code == 422
    items = {item["index"]: item for item in resp.json()["detail"]["items"]}
    assert set(items) == {0, 1}
    assert items[0]["batch_conflicts"][0]["index"] == 1
    assert items[0]["batch_conflicts"][0]["reason"] == "confusable"
    assert items[1]["batch_conflicts"][0]["index"] == 0
    assert client.get("/namespaces/chat/names").json() == []


def test_batch_db_conflict_returns_422_with_conflict_records(client):
    assert register(client, "app").status_code == 201

    resp = register_batch(client, ["hello", CYRILLIC_APP])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert [item["index"] for item in detail["items"]] == [1]
    conflict = detail["items"][0]["conflicts"][0]
    assert conflict["original_name"] == "app"
    assert conflict["reason"] == "confusable"

    # 只有预先注册的 app，批次未写入
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["app"]


def test_batch_mixed_script_returns_422_with_script_details(client):
    resp = register_batch(client, ["clean", MIXED_ADMIN])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert [item["index"] for item in detail["items"]] == [1]
    item = detail["items"][0]
    assert item["mixed_script_violation"] is True
    assert item["restricted_scripts_used"] == ["Cyrillic", "Latin"]
    by_script = {d["script"]: d for d in item["script_details"]}
    assert by_script["Cyrillic"]["code_points"] == ["U+043C"]
    assert client.get("/namespaces/chat/names").json() == []


def test_batch_aggregates_script_issues_and_all_conflicts(client):
    assert register(client, "Alice").status_code == 201

    # 一批同时覆盖：库内冲突、批内冲突、混脚本
    resp = register_batch(client, ["ALICE", "app", CYRILLIC_APP, MIXED_ADMIN])
    assert resp.status_code == 422
    items = {item["index"]: item for item in resp.json()["detail"]["items"]}
    assert set(items) == {0, 1, 2, 3}
    assert items[0]["conflicts"][0]["reason"] == "normalized_match"  # 与库内 Alice
    assert items[1]["batch_conflicts"][0]["index"] == 2  # 批内同形
    assert items[2]["batch_conflicts"][0]["index"] == 1
    assert items[3]["mixed_script_violation"] is True

    # 一条都不落库
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["Alice"]


def test_batch_concurrent_race_returns_409_with_refreshed_conflicts(client, monkeypatch):
    """并发竞争：批次事务开始时另一连接已抢先写入冲突名称。

    insert_batch 事务内复查发现判定失效 -> 回滚 -> 重新分析 -> 409。
    """
    real_insert_batch = service.insert_batch

    def racing_insert_batch(conn, namespace, names):
        # 模拟并发写入：另一个连接在批次事务前抢注 "Alice"
        other = database.get_connection()
        try:
            service.insert_name(other, namespace, "Alice", "alice")
        finally:
            other.close()
        return real_insert_batch(conn, namespace, names)

    monkeypatch.setattr(service, "insert_batch", racing_insert_batch)

    resp = register_batch(client, ["alice", "bob"])
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "name_conflict"
    assert [item["index"] for item in detail["items"]] == [0]
    assert detail["items"][0]["conflicts"][0]["original_name"] == "Alice"
    assert detail["items"][0]["conflicts"][0]["reason"] == "normalized_match"

    # 批次整体回滚：库内只有并发写入的 Alice
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["Alice"]


def test_insert_batch_rolls_back_when_conflict_appears_in_transaction(client):
    """服务层兜底：事务内复查发现库内冲突时整批回滚。"""
    conn = database.get_connection()
    try:
        service.insert_name(conn, "chat", "app", "app")
        with pytest.raises(service.NameConflictError):
            service.insert_batch(conn, "chat", ["alice", CYRILLIC_APP])
        # alice 未写入，整批回滚
        assert [r["original_name"] for r in service.list_names(conn, "chat")] == ["app"]
    finally:
        conn.close()


def test_insert_batch_unique_constraint_rolls_back_entire_batch(client):
    """服务层兜底：批内规范名重复触发唯一约束时整批回滚。"""
    conn = database.get_connection()
    try:
        with pytest.raises(service.NameConflictError):
            service.insert_batch(conn, "chat", ["dup", "DUP"])
        assert service.list_names(conn, "chat") == []
    finally:
        conn.close()


def test_batch_concurrent_requests_single_winner_no_5xx(client):
    """并发回归：同一批次并发提交，恰好一个 201，其余 409/422，无 5xx。

    同时覆盖连接跨线程使用（check_same_thread=False）的回归。
    """

    def post(_):
        return client.post(
            "/namespaces/chat/names/batch", json={"names": ["n1", "n2"]}
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(post, range(8)))

    assert not any(code >= 500 for code in codes)
    assert codes.count(201) == 1
    assert set(codes) <= {201, 409, 422}
    # 只有一个批次落库（all-or-nothing）
    assert len(client.get("/namespaces/chat/names").json()) == 2


# ---------- 预占 POST /namespaces/{namespace}/reservations ----------


def reserve(client, name, ttl_seconds=120, namespace="chat"):
    return client.post(
        f"/namespaces/{namespace}/reservations",
        json={"name": name, "ttl_seconds": ttl_seconds},
    )


def claim(client, token, namespace="chat"):
    return client.post(
        f"/namespaces/{namespace}/reservations/{token}/claim"
    )


def test_reserve_returns_201_with_unguessable_token_and_expires_at(client):
    resp = reserve(client, "Alice", ttl_seconds=60)
    assert resp.status_code == 201
    body = resp.json()
    assert body["namespace"] == "chat"
    assert body["original_name"] == "Alice"
    assert body["normalized_name"] == "alice"
    assert body["ttl_seconds"] == 60
    # token 不可猜测：32 字节随机 -> urlsafe base64 至少 40 字符
    assert len(body["token"]) >= 40
    assert body["expires_at"].endswith("Z")
    assert body["id"] >= 1

    # 两个独立预占的 token 必须不同
    other = reserve(client, "Bob")
    assert other.status_code == 201
    assert other.json()["token"] != body["token"]

    # 预占期间名称尚未注册
    assert client.get("/namespaces/chat/names").json() == []


def test_reserve_ttl_bounds_validated(client):
    assert reserve(client, "a", ttl_seconds=30).status_code == 201
    assert reserve(client, "b", ttl_seconds=600).status_code == 201
    assert reserve(client, "c", ttl_seconds=29).status_code == 422
    assert reserve(client, "d", ttl_seconds=601).status_code == 422
    assert reserve(client, "e", ttl_seconds=0).status_code == 422


def test_reserve_mixed_script_returns_422_and_does_not_persist(client):
    resp = reserve(client, MIXED_ADMIN)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "mixed_script_violation"
    assert detail["restricted_scripts_used"] == ["Cyrillic", "Latin"]

    conn = database.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM name_reservations").fetchone()[0] == 0
    finally:
        conn.close()


def test_active_reservation_blocks_single_register_normalized(client):
    r = reserve(client, "Alice").json()

    # 大小写折叠后相同 -> 409，冲突来源标记为 reservation
    resp = register(client, "ALICE")
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["reason"] == "normalized_match"
    assert conflict["source"] == "reservation"
    assert conflict["original_name"] == "Alice"
    assert conflict["expires_at"] == r["expires_at"]

    # 注册没有落库
    assert client.get("/namespaces/chat/names").json() == []


def test_active_reservation_blocks_single_register_confusable(client):
    reserve(client, "app")

    resp = register(client, CYRILLIC_APP)
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["reason"] == "confusable"
    assert conflict["source"] == "reservation"


def test_active_reservation_blocks_new_reservation(client):
    reserve(client, "app")

    resp = reserve(client, CYRILLIC_APP)
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["source"] == "reservation"
    assert conflict["reason"] == "confusable"


def test_active_reservation_blocks_batch_register(client):
    reserve(client, "app")

    resp = register_batch(client, ["hello", CYRILLIC_APP])
    assert resp.status_code == 422
    items = resp.json()["detail"]["items"]
    assert items[0]["index"] == 1
    assert items[0]["conflicts"][0]["source"] == "reservation"
    assert items[0]["conflicts"][0]["reason"] == "confusable"
    # 整批不落库，预占也未被消费
    assert client.get("/namespaces/chat/names").json() == []


def test_registered_name_blocks_reservation_with_registered_source(client):
    assert register(client, "app").status_code == 201

    resp = reserve(client, CYRILLIC_APP)
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["source"] == "registered"
    assert conflict["reason"] == "confusable"
    assert "expires_at" not in conflict


def test_check_reports_active_reservation_conflict(client):
    reserve(client, "app")

    resp = client.post("/namespaces/chat/check", json={"name": CYRILLIC_APP})
    assert resp.status_code == 200
    conflicts = resp.json()["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["source"] == "reservation"
    assert conflicts[0]["reason"] == "confusable"


def test_reservations_are_namespace_isolated(client):
    reserve(client, "app", namespace="a")
    assert register(client, "app", namespace="b").status_code == 201
    assert reserve(client, CYRILLIC_APP, namespace="b").status_code == 409
    # 命名空间 a 中预占仍然有效
    assert register(client, "app", namespace="a").status_code == 409


# ---------- 确认注册 POST .../reservations/{token}/claim ----------


def test_claim_success_writes_display_name_and_marks_consumed(client):
    r = reserve(client, "Alice", ttl_seconds=60).json()

    resp = claim(client, r["token"])
    assert resp.status_code == 201
    body = resp.json()
    assert body["original_name"] == "Alice"
    assert body["normalized_name"] == "alice"
    assert body["namespace"] == "chat"
    assert body["token"] == r["token"]
    assert body["expires_at"] == r["expires_at"]
    assert body["consumed_at"].endswith("Z")
    assert body["id"] >= 1

    # 名称已真正注册
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["Alice"]


def test_claim_unknown_token_returns_404(client):
    resp = claim(client, "does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "reservation_not_found"


def test_claim_token_from_other_namespace_returns_404(client):
    r = reserve(client, "app", namespace="a").json()
    resp = claim(client, r["token"], namespace="b")
    assert resp.status_code == 404
    # 预占仍可在原命名空间确认
    assert claim(client, r["token"], namespace="a").status_code == 201


def test_claim_twice_only_first_succeeds(client):
    r = reserve(client, "Alice").json()

    assert claim(client, r["token"]).status_code == 201
    second = claim(client, r["token"])
    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "reservation_already_claimed"

    # 只注册了一次
    assert len(client.get("/namespaces/chat/names").json()) == 1


def test_claim_concurrent_requests_exactly_one_wins(client):
    r = reserve(client, "Alice").json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: claim(client, r["token"]), range(8)))

    codes = [resp.status_code for resp in responses]
    assert codes.count(201) == 1
    assert set(codes) <= {201, 409}
    assert not any(code >= 500 for code in codes)
    assert len(client.get("/namespaces/chat/names").json()) == 1


def test_claim_expired_returns_410_and_lazily_deletes(client):
    r = reserve(client, "Alice", ttl_seconds=30).json()

    # 直接把 expires_at 改成过去时刻，模拟 TTL 到期
    conn = database.get_connection()
    try:
        conn.execute(
            "UPDATE name_reservations SET expires_at = '2000-01-01T00:00:00.000Z'"
        )
        conn.commit()
    finally:
        conn.close()

    resp = claim(client, r["token"])
    assert resp.status_code == 410
    assert resp.json()["detail"]["error"] == "reservation_expired"

    # 惰性清理：过期记录已删除，名称可被重新注册
    conn = database.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM name_reservations").fetchone()[0] == 0
    finally:
        conn.close()
    assert register(client, "Alice").status_code == 201


def test_expired_reservation_does_not_block_and_is_purged_on_next_write(client):
    reserve(client, "app", ttl_seconds=30)
    conn = database.get_connection()
    try:
        conn.execute(
            "UPDATE name_reservations SET expires_at = '2000-01-01T00:00:00.000Z'"
        )
        conn.commit()
    finally:
        conn.close()

    # 过期预占不阻止新预占；创建时惰性清理旧记录
    resp = reserve(client, "app")
    assert resp.status_code == 201
    conn = database.get_connection()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM name_reservations WHERE original_name = 'app'"
        ).fetchone()[0]
        assert count == 1  # 旧的已删除，只剩新的
    finally:
        conn.close()


def test_claim_conflict_with_registered_name_returns_409_and_writes_nothing(client):
    r = reserve(client, "app").json()
    # 预占期间另一个名字（不与预占冲突的）被注册后，再注册与预占同规范名的名字
    # —— 单条注册接口会被预占挡住，因此这里直接走服务层制造并发抢注
    conn = database.get_connection()
    try:
        service.insert_name(conn, "chat", "APP", "app")
    finally:
        conn.close()

    resp = claim(client, r["token"])
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "name_conflict"
    assert detail["conflicts"][0]["original_name"] == "APP"
    assert detail["conflicts"][0]["source"] == "registered"

    # 冲突不写入，预占未被消费（但名称仍被注册方占着，后续确认依旧失败）
    names = client.get("/namespaces/chat/names").json()
    assert [n["original_name"] for n in names] == ["APP"]
    assert claim(client, r["token"]).status_code == 409


def test_claim_conflict_with_other_reservation_returns_409(client):
    # 两个互相同形的预占不能同时存在：先预占 app，再用同形名会被 409。
    # 因此冲突场景用“注册名 vs 预占”之外的手段：同规范名预占经直接插入制造，
    # 模拟并发下两笔预占都通过了各自的事务前分析。
    first = reserve(client, "app").json()
    conn = database.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO name_reservations
                (namespace, original_name, normalized_name, token, expires_at)
            VALUES ('chat', 'APP', 'app', ?, '2999-01-01T00:00:00.000Z')
            """,
            ("second-conflicting-token",),
        )
        conn.commit()
    finally:
        conn.close()

    resp = claim(client, first["token"])
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["source"] == "reservation"
    assert conflict["original_name"] == "APP"
    assert client.get("/namespaces/chat/names").json() == []


def test_claim_after_success_name_blocks_others(client):
    r = reserve(client, "app").json()
    assert claim(client, r["token"]).status_code == 201

    # 确认后与普通注册完全等价：同形名 409
    assert register(client, CYRILLIC_APP).status_code == 409
    assert reserve(client, CYRILLIC_APP).status_code == 409
    # 已消费预占不再出现在冲突明细中
    resp = client.post("/namespaces/chat/check", json={"name": CYRILLIC_APP})
    assert resp.json()["conflicts"][0]["source"] == "registered"


def test_service_claim_rolls_back_when_conflict_detected_in_transaction(client):
    """服务层：claim 事务内复查发现冲突时抛 NameConflictError 且不写入。"""
    r = reserve(client, "app").json()

    # 直接制造与预占同规范名的已注册记录（绕过 API 的预占拦截）
    conn = database.get_connection()
    try:
        service.insert_name(conn, "chat", "APP", "app")
    finally:
        conn.close()

    claim_conn = database.get_connection()
    try:
        with pytest.raises(service.NameConflictError) as exc_info:
            service.claim_reservation(claim_conn, "chat", r["token"])
        conflicts = exc_info.value.conflicts
        assert conflicts and conflicts[0]["source"] == "registered"

        # 事务已回滚：预占仍是未消费状态，可重复进入确认（依旧冲突）
        row = claim_conn.execute(
            "SELECT consumed_at FROM name_reservations WHERE token = ?", (r["token"],)
        ).fetchone()
        assert row["consumed_at"] is None
    finally:
        claim_conn.close()


def test_reserve_concurrent_race_returns_409_with_refreshed_conflicts(client, monkeypatch):
    """并发竞争：只读分析通过后、写事务复查前另一连接抢注同名 -> 409，不落预占。"""
    real_create_reservation = service.create_reservation

    def racing_create_reservation(conn, namespace, name, ttl_seconds):
        # 模拟并发写入：另一个连接在预占写事务前抢注 "Alice"
        other = database.get_connection()
        try:
            service.insert_name(other, namespace, "Alice", "alice")
        finally:
            other.close()
        return real_create_reservation(conn, namespace, name, ttl_seconds)

    monkeypatch.setattr(service, "create_reservation", racing_create_reservation)

    resp = reserve(client, "Alice")
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["original_name"] == "Alice"
    assert conflict["source"] == "registered"

    conn = database.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM name_reservations").fetchone()[0] == 0
    finally:
        conn.close()


# ---------- 合并命名空间 POST /namespaces/{source}/merge ----------


def merge(client, source="a", target="b"):
    return client.post(f"/namespaces/{source}/merge", json={"target_namespace": target})


def test_merge_success_migrates_names_and_reservations(client):
    assert register(client, "alice", namespace="a").status_code == 201
    assert register(client, "Ｂｏｂ", namespace="a").status_code == 201
    active = reserve(client, "carol", namespace="a").json()
    consumed = reserve(client, "dave", namespace="a").json()
    assert claim(client, consumed["token"], namespace="a").status_code == 201
    # 目标命名空间已有的记录不受影响
    assert register(client, "erin", namespace="b").status_code == 201

    resp = merge(client, "a", "b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_namespace"] == "a"
    assert body["target_namespace"] == "b"
    assert body["merged_names"] == 3  # alice、Ｂｏｂ、dave（claim 后注册）
    assert body["merged_reservations"] == 2  # 有效预占 carol + 已消费历史记录 dave

    # 源命名空间已清空，目标命名空间包含全部名称（id 与原记录保持不变）
    assert client.get("/namespaces/a/names").json() == []
    names = {n["original_name"]: n for n in client.get("/namespaces/b/names").json()}
    assert set(names) == {"alice", "Ｂｏｂ", "dave", "erin"}
    assert names["Ｂｏｂ"]["normalized_name"] == "bob"

    # 迁移后的有效预占在目标命名空间继续占位，并可在目标命名空间确认
    resp = register(client, "CAROL", namespace="b")
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflicts"][0]["source"] == "reservation"
    assert claim(client, active["token"], namespace="a").status_code == 404
    assert claim(client, active["token"], namespace="b").status_code == 201

    # 源命名空间已空：同名可以重新注册
    assert register(client, "alice", namespace="a").status_code == 201


def test_merge_empty_source_returns_zero_counts(client):
    assert register(client, "erin", namespace="b").status_code == 201

    resp = merge(client, "no-such-namespace", "b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["merged_names"] == 0
    assert body["merged_reservations"] == 0
    # 目标命名空间不受影响
    assert len(client.get("/namespaces/b/names").json()) == 1


def test_merge_normalized_conflict_returns_409_with_both_records(client):
    assert register(client, "Alice", namespace="a").status_code == 201
    assert register(client, "ALICE", namespace="b").status_code == 201

    resp = merge(client, "a", "b")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "merge_conflict"
    assert detail["source_namespace"] == "a"
    assert detail["target_namespace"] == "b"
    pair = detail["conflicts"][0]
    assert pair["reason"] == "normalized_match"
    assert pair["source_record"]["original_name"] == "Alice"
    assert pair["source_record"]["kind"] == "registered"
    assert pair["target_record"]["original_name"] == "ALICE"
    assert pair["target_record"]["kind"] == "registered"

    # 409 不修改任何数据
    assert [n["original_name"] for n in client.get("/namespaces/a/names").json()] == ["Alice"]
    assert [n["original_name"] for n in client.get("/namespaces/b/names").json()] == ["ALICE"]


def test_merge_confusable_conflict_returns_409(client):
    assert register(client, "app", namespace="a").status_code == 201
    assert register(client, CYRILLIC_APP, namespace="b").status_code == 201

    resp = merge(client, "a", "b")
    assert resp.status_code == 409
    pair = resp.json()["detail"]["conflicts"][0]
    assert pair["reason"] == "confusable"
    assert pair["source_record"]["normalized_name"] == "app"
    assert pair["target_record"]["normalized_name"] == CYRILLIC_APP
    assert len(client.get("/namespaces/a/names").json()) == 1
    assert len(client.get("/namespaces/b/names").json()) == 1


def test_merge_conflict_with_active_reservation_on_either_side(client):
    # 源侧有效预占 vs 目标侧已注册名称
    r = reserve(client, "app", namespace="a").json()
    assert register(client, CYRILLIC_APP, namespace="b").status_code == 201

    resp = merge(client, "a", "b")
    assert resp.status_code == 409
    pair = resp.json()["detail"]["conflicts"][0]
    assert pair["reason"] == "confusable"
    assert pair["source_record"]["kind"] == "reservation"
    assert pair["source_record"]["expires_at"] == r["expires_at"]
    assert pair["target_record"]["kind"] == "registered"

    # 反向同样成立：目标侧有效预占 vs 源侧已注册名称
    assert register(client, "app", namespace="c").status_code == 201
    r2 = reserve(client, CYRILLIC_APP, namespace="d").json()
    resp = merge(client, "c", "d")
    assert resp.status_code == 409
    pair = resp.json()["detail"]["conflicts"][0]
    assert pair["source_record"]["kind"] == "registered"
    assert pair["target_record"]["kind"] == "reservation"
    assert pair["target_record"]["expires_at"] == r2["expires_at"]

    # 两侧数据均未变化
    assert client.get("/namespaces/c/names").json()[0]["original_name"] == "app"


def test_merge_expired_reservation_does_not_block_and_is_purged(client):
    reserve(client, "app", ttl_seconds=30, namespace="a")
    conn = database.get_connection()
    try:
        conn.execute(
            "UPDATE name_reservations SET expires_at = '2000-01-01T00:00:00.000Z'"
        )
        conn.commit()
    finally:
        conn.close()
    # 过期预占不参与冲突判定：目标侧的同形名不阻止合并
    assert register(client, CYRILLIC_APP, namespace="b").status_code == 201

    resp = merge(client, "a", "b")
    assert resp.status_code == 200
    assert resp.json()["merged_reservations"] == 0  # 过期记录被惰性清理，不迁移

    conn = database.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM name_reservations").fetchone()[0] == 0
    finally:
        conn.close()


def test_merge_same_namespace_rejected(client):
    resp = merge(client, "a", "a")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_merge_target"


def test_merge_blank_or_missing_target_rejected(client):
    assert client.post(
        "/namespaces/a/merge", json={"target_namespace": "   "}
    ).status_code == 422
    assert client.post("/namespaces/a/merge", json={}).status_code == 422


def test_merge_concurrent_race_returns_409_with_refreshed_conflicts(client, monkeypatch):
    """并发竞争：合并事务开始前另一连接在目标命名空间抢注同规范名 -> 409，不迁移。"""
    assert register(client, "alice", namespace="a").status_code == 201
    real_merge_namespaces = service.merge_namespaces

    def racing_merge_namespaces(conn, source_namespace, target_namespace):
        # 模拟并发写入：另一个连接在合并事务前向目标命名空间抢注 "ALICE"
        other = database.get_connection()
        try:
            service.insert_name(other, target_namespace, "ALICE", "alice")
        finally:
            other.close()
        return real_merge_namespaces(conn, source_namespace, target_namespace)

    monkeypatch.setattr(service, "merge_namespaces", racing_merge_namespaces)

    resp = merge(client, "a", "b")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "merge_conflict"
    pair = detail["conflicts"][0]
    assert pair["reason"] == "normalized_match"
    assert pair["source_record"]["original_name"] == "alice"
    assert pair["target_record"]["original_name"] == "ALICE"

    # 整体回滚：源侧记录未迁移，目标侧只有并发写入的 ALICE
    assert [n["original_name"] for n in client.get("/namespaces/a/names").json()] == ["alice"]
    assert [n["original_name"] for n in client.get("/namespaces/b/names").json()] == ["ALICE"]


def test_merge_unique_constraint_failure_rolls_back_everything(client, monkeypatch):
    """服务层兜底：唯一约束冲突时整体回滚，重新检测后返回 409。"""
    assert register(client, "alice", namespace="a").status_code == 201
    assert register(client, "ALICE", namespace="b").status_code == 201
    # 只绕过事务内的第一次复查，让 UPDATE 触发 (namespace, normalized_name)
    # 唯一约束；回滚后的重新检测仍走真实逻辑
    real_find_merge_conflicts = service.find_merge_conflicts
    calls = {"count": 0}

    def skip_first_find_merge_conflicts(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return real_find_merge_conflicts(*args, **kwargs)

    monkeypatch.setattr(service, "find_merge_conflicts", skip_first_find_merge_conflicts)

    resp = merge(client, "a", "b")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "merge_conflict"
    # 回滚后重新检测到的冲突明细仍带双方记录
    assert detail["conflicts"][0]["source_record"]["original_name"] == "alice"
    assert detail["conflicts"][0]["target_record"]["original_name"] == "ALICE"

    # 两侧数据均未变化
    assert [n["original_name"] for n in client.get("/namespaces/a/names").json()] == ["alice"]
    assert [n["original_name"] for n in client.get("/namespaces/b/names").json()] == ["ALICE"]


def test_service_merge_rolls_back_when_conflict_detected_in_transaction(client):
    """服务层：merge 事务内复查发现冲突时抛 MergeConflictError 且不迁移。"""
    conn = database.get_connection()
    try:
        service.insert_name(conn, "a", "app", "app")
        service.insert_name(conn, "b", CYRILLIC_APP, CYRILLIC_APP)
        with pytest.raises(service.MergeConflictError) as exc_info:
            service.merge_namespaces(conn, "a", "b")
        conflicts = exc_info.value.conflicts
        assert conflicts[0]["reason"] == "confusable"
        assert conflicts[0]["source_record"]["kind"] == "registered"

        # 事务已回滚：两侧记录保持原样
        assert [r["original_name"] for r in service.list_names(conn, "a")] == ["app"]
        assert [r["original_name"] for r in service.list_names(conn, "b")] == [CYRILLIC_APP]
    finally:
        conn.close()


def test_merge_concurrent_requests_no_5xx_and_all_names_merged_once(client):
    """并发回归：同一源命名空间并发合并，名称恰好迁移一次，无 5xx。"""
    assert register_batch(client, ["n1", "n2"], namespace="a").status_code == 201

    def post(_):
        return client.post("/namespaces/a/merge", json={"target_namespace": "b"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(post, range(8)))

    codes = [resp.status_code for resp in responses]
    assert not any(code >= 500 for code in codes)
    assert set(codes) <= {200, 409}
    # 名称恰好迁移一次：各次合并报告的迁移数量之和等于源侧原有数量
    assert sum(resp.json()["merged_names"] for resp in responses if resp.status_code == 200) == 2
    assert client.get("/namespaces/a/names").json() == []
    assert len(client.get("/namespaces/b/names").json()) == 2


def test_openapi_documents_merge_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/namespaces/{source}/merge" in paths
    operation = paths["/namespaces/{source}/merge"]["post"]
    assert "409" in operation["responses"]
    assert "422" in operation["responses"]


# ---------- 原子改名 PATCH /namespaces/{namespace}/names/{id} ----------


def rename(client, name_id, name, expected_revision, namespace="chat"):
    return client.patch(
        f"/namespaces/{namespace}/names/{name_id}",
        json={"name": name, "expected_revision": expected_revision},
    )


def test_rename_success_updates_names_and_bumps_revision_keeps_id_and_created_at(client):
    created = register(client, "Alice").json()
    name_id = created["id"]
    assert created["revision"] == 1

    resp = rename(client, name_id, "Ｂｏｂ", expected_revision=1)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == name_id  # id 保留
    assert body["namespace"] == "chat"
    assert body["original_name"] == "Ｂｏｂ"  # 原名原样更新
    assert body["normalized_name"] == "bob"  # 规范名重新计算
    assert body["revision"] == 2  # revision 递增
    assert body["created_at"] == created["created_at"]  # created_at 保留

    # 再次改名：revision 继续递增，规范名查重按新名称生效
    resp = rename(client, name_id, "Charlie", expected_revision=2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == name_id
    assert body["original_name"] == "Charlie"
    assert body["normalized_name"] == "charlie"
    assert body["revision"] == 3
    assert body["created_at"] == created["created_at"]

    # 列表接口反映改名结果（仍只有一条记录），旧规范名释放
    names = client.get("/namespaces/chat/names").json()
    assert len(names) == 1
    assert names[0]["original_name"] == "Charlie"
    assert names[0]["revision"] == 3
    freed = register(client, "ALICE")  # 旧规范名 alice 可被重新注册
    assert freed.status_code == 201
    assert freed.json()["id"] != name_id


def test_rename_to_same_name_succeeds_and_bumps_revision(client):
    created = register(client, "Alice").json()

    # 改成与当前完全相同的名称：排除自身后无冲突，允许并递增 revision
    resp = rename(client, created["id"], "Alice", expected_revision=1)
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2
    assert resp.json()["normalized_name"] == "alice"


def test_rename_stale_revision_returns_409_with_current_revision(client):
    created = register(client, "Alice").json()
    name_id = created["id"]
    assert rename(client, name_id, "Alicia", expected_revision=1).status_code == 200

    # 持旧版本再次改名 -> 409 stale_revision，附当前 revision
    resp = rename(client, name_id, "Bob", expected_revision=1)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "stale_revision"
    assert detail["current_revision"] == 2

    # 改名未生效
    assert client.get("/namespaces/chat/names").json()[0]["original_name"] == "Alicia"

    # 用最新版本可成功
    resp = rename(client, name_id, "Bob", expected_revision=2)
    assert resp.status_code == 200
    assert resp.json()["revision"] == 3


def test_rename_unknown_id_returns_404(client):
    resp = rename(client, 999, "whatever", expected_revision=1)
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "name_not_found"


def test_rename_id_from_other_namespace_returns_404(client):
    created = register(client, "Alice", namespace="a").json()

    # id 属于命名空间 a，在 b 下改名 -> 404（只能改本空间记录）
    resp = rename(client, created["id"], "Bob", expected_revision=1, namespace="b")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "name_not_found"

    # 原空间记录未被修改
    assert register(client, "alice", namespace="a").status_code == 409
    names = client.get("/namespaces/a/names").json()
    assert names[0]["original_name"] == "Alice"
    assert names[0]["revision"] == 1


def test_rename_normalized_conflict_returns_409_with_existing_record(client):
    first = register(client, "Alice").json()
    other = register(client, "Bob").json()

    # 把 Bob 改成与 Alice 规范名相同 -> 409，冲突明细指向 Alice（排除自身）
    resp = rename(client, other["id"], "ALICE", expected_revision=1)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "name_conflict"
    assert detail["normalized_name"] == "alice"
    assert len(detail["conflicts"]) == 1
    conflict = detail["conflicts"][0]
    assert conflict["id"] == first["id"]
    assert conflict["original_name"] == "Alice"
    assert conflict["reason"] == "normalized_match"
    assert conflict["source"] == "registered"

    # 改名未生效，revision 不递增
    unchanged = client.get("/namespaces/chat/names").json()
    bob = next(n for n in unchanged if n["id"] == other["id"])
    assert bob["original_name"] == "Bob"
    assert bob["revision"] == 1


def test_rename_confusable_conflict_returns_409(client):
    registered = register(client, "app").json()
    target = register(client, "hello").json()

    # 把 hello 改成纯西里尔 арр：与 app 同形但不混脚本 -> 409 confusable
    resp = rename(client, target["id"], CYRILLIC_APP, expected_revision=1)
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["id"] == registered["id"]
    assert conflict["reason"] == "confusable"
    assert client.get("/namespaces/chat/names").json()[1]["original_name"] == "hello"


def test_rename_does_not_conflict_with_itself(client):
    created = register(client, "Alice").json()

    # 仅大小写不同的新规范名与自身相等：排除自身后无冲突
    resp = rename(client, created["id"], "ALICE", expected_revision=1)
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_name"] == "ALICE"
    assert body["normalized_name"] == "alice"
    assert body["revision"] == 2


def test_rename_active_reservation_blocks_even_when_registered_name_freed(client):
    created = register(client, "Alice").json()
    # 一条有效预占占用 carol
    reservation = reserve(client, "carol", ttl_seconds=120).json()

    # 改名到预占名：预占不排除 -> 409，冲突来源为 reservation
    resp = rename(client, created["id"], "CAROL", expected_revision=1)
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["source"] == "reservation"
    assert conflict["reason"] == "normalized_match"
    assert conflict["id"] == reservation["id"]
    assert conflict["expires_at"] == reservation["expires_at"]
    assert client.get("/namespaces/chat/names").json()[0]["revision"] == 1


def test_rename_mixed_script_returns_422_with_script_details(client):
    created = register(client, "Alice").json()

    resp = rename(client, created["id"], MIXED_ADMIN, expected_revision=1)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "mixed_script_violation"
    assert detail["restricted_scripts_used"] == ["Cyrillic", "Latin"]
    by_script = {d["script"]: d for d in detail["script_details"]}
    assert by_script["Cyrillic"]["code_points"] == ["U+043C"]

    # 422 不修改记录，revision 不递增
    record = client.get("/namespaces/chat/names").json()[0]
    assert record["original_name"] == "Alice"
    assert record["revision"] == 1


def test_rename_blank_name_or_non_positive_revision_rejected(client):
    created = register(client, "Alice").json()
    assert rename(client, created["id"], "   ", expected_revision=1).status_code == 422
    assert rename(client, created["id"], "Bob", expected_revision=0).status_code == 422
    # 缺少 expected_revision / name 同样 422
    resp = client.patch(
        f"/namespaces/chat/names/{created['id']}", json={"name": "Bob"}
    )
    assert resp.status_code == 422


def test_rename_non_integer_id_returns_422(client):
    resp = client.patch(
        "/namespaces/chat/names/not-an-int",
        json={"name": "Bob", "expected_revision": 1},
    )
    assert resp.status_code == 422


def test_rename_concurrent_requests_exactly_one_wins(client):
    """并发改名：版本校验串行化，恰好一次成功，其余 409 stale_revision，无 5xx。"""
    created = register(client, "Alice").json()
    name_id = created["id"]

    def post(seq):
        suffix = chr(ord("a") + seq)
        return rename(client, name_id, f"name-{suffix}", expected_revision=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(post, range(8)))

    codes = [resp.status_code for resp in responses]
    assert codes.count(200) == 1
    assert set(codes) <= {200, 409}
    assert not any(code >= 500 for code in codes)

    record = client.get("/namespaces/chat/names").json()[0]
    assert record["revision"] == 2  # 只有一次改名生效
    # 失败方拿到的 current_revision 都是 2，可用最新版本重试成功
    stale = [resp for resp in responses if resp.status_code == 409]
    assert all(resp.json()["detail"]["error"] == "stale_revision" for resp in stale)
    assert all(resp.json()["detail"]["current_revision"] == 2 for resp in stale)


def test_rename_concurrent_race_returns_409_with_current_revision(client, monkeypatch):
    """并发竞争：改名事务开始前另一连接已抢先改名 -> 409 stale_revision，不修改。"""
    created = register(client, "Alice").json()
    name_id = created["id"]
    real_rename_name = service.rename_name

    def racing_rename_name(conn, namespace, target_id, name, expected_revision):
        # 模拟并发改名：另一个连接在本事务前把 revision 提升到 2
        other = database.get_connection()
        try:
            other.execute(
                """
                UPDATE display_names
                   SET original_name = 'Racer', normalized_name = 'racer',
                       revision = revision + 1
                 WHERE id = ?
                """,
                (target_id,),
            )
            other.commit()
        finally:
            other.close()
        return real_rename_name(conn, namespace, target_id, name, expected_revision)

    monkeypatch.setattr(service, "rename_name", racing_rename_name)

    resp = rename(client, name_id, "Bob", expected_revision=1)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "stale_revision"
    assert detail["current_revision"] == 2

    # 抢先改名生效，本次请求未覆盖
    record = client.get("/namespaces/chat/names").json()[0]
    assert record["original_name"] == "Racer"
    assert record["revision"] == 2


def test_service_rename_is_atomic_in_single_immediate_transaction(client):
    """服务层：各失败分支都在同一事务内回滚，成功分支保留 id/created_at。"""
    conn = database.get_connection()
    try:
        record = service.register_name(conn, "chat", "Alice", "alice")
        name_id = record["id"]
        created_at = record["created_at"]

        # 预置另一条名称制造冲突
        service.insert_name(conn, "chat", "bob", "bob")

        # 版本过期
        with pytest.raises(service.StaleRevisionError):
            service.rename_name(conn, "chat", name_id, "x", expected_revision=99)
        # 混脚本
        with pytest.raises(service.MixedScriptNameError):
            service.rename_name(conn, "chat", name_id, MIXED_ADMIN, expected_revision=1)
        # 冲突（排除自身后与 bob 同规范名）
        with pytest.raises(service.NameConflictError) as exc_info:
            service.rename_name(conn, "chat", name_id, "BOB", expected_revision=1)
        assert exc_info.value.conflicts[0]["reason"] == "normalized_match"
        # 不存在 / 跨命名空间
        with pytest.raises(service.NameNotFoundError):
            service.rename_name(conn, "chat", 999, "x", expected_revision=1)
        with pytest.raises(service.NameNotFoundError):
            service.rename_name(conn, "other", name_id, "x", expected_revision=1)

        # 全部失败后记录原样
        rows = service.list_names(conn, "chat")
        assert [r["original_name"] for r in rows] == ["Alice", "bob"]
        assert rows[0]["revision"] == 1

        # 成功路径
        updated = service.rename_name(
            conn, "chat", name_id, "Alicia", expected_revision=1
        )
        assert updated["id"] == name_id
        assert updated["created_at"] == created_at
        assert updated["original_name"] == "Alicia"
        assert updated["normalized_name"] == "alicia"
        assert updated["revision"] == 2
    finally:
        conn.close()


def test_init_db_backfills_revision_on_legacy_database(tmp_path):
    """启动幂等迁移：老版本 display_names 无 revision 列时自动补齐为 1。"""
    db_path = tmp_path / "legacy.db"
    import sqlite3

    # 直接以老结构建库并写入两条记录
    with sqlite3.connect(db_path) as legacy:
        legacy.execute(
            """
            CREATE TABLE display_names (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace       TEXT NOT NULL,
                original_name   TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT '',
                UNIQUE (namespace, normalized_name)
            )
            """
        )
        legacy.execute(
            "INSERT INTO display_names (namespace, original_name, normalized_name) "
            "VALUES ('chat', 'Alice', 'alice')"
        )
        legacy.execute(
            "INSERT INTO display_names (namespace, original_name, normalized_name) "
            "VALUES ('chat', 'Bob', 'bob')"
        )
        legacy.commit()

    # 启动迁移（幂等：连续执行两次）
    database.init_db(str(db_path))
    database.init_db(str(db_path))

    with sqlite3.connect(db_path) as migrated:
        cols = {row[1] for row in migrated.execute("PRAGMA table_info(display_names)")}
        assert "revision" in cols
        rows = migrated.execute(
            "SELECT id, revision, original_name FROM display_names ORDER BY id"
        ).fetchall()
        assert rows == [(1, 1, "Alice"), (2, 1, "Bob")]


def test_openapi_documents_rename_endpoint(client):
    operation = client.get("/openapi.json").json()["paths"][
        "/namespaces/{namespace}/names/{name_id}"
    ]["patch"]
    assert "200" in operation["responses"]
    assert "404" in operation["responses"]
    assert "409" in operation["responses"]
    assert "422" in operation["responses"]
