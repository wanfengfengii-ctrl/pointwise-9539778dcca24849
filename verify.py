"""snapshots.Store 核验套件:

1. 语义单元测试(题目示例、四类操作、原子性、隔离性、约束上限等);
2. 独立参照核验:用一个简单的「深拷贝 + 普通 list/dict」参照模型,
   对随机操作序列做差分比对(结果、报错行为、版本号须完全一致);
3. 长数组头部插删资源实测:每次操作耗时随数组长度的变化、单次插入
   新增节点数(结构共享证据)、500 次头插的内存增量。

退出码 0 表示全部通过,1 表示有失败。
"""

import copy
import math
import random
import sys
import time
import traceback
import tracemalloc

import snapshots
from snapshots import Store


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError from %r%r" % (fn, args))


def check_avl(node):
    """白盒校验 AVL 不变量,返回 (height, size)。"""
    if node is None:
        return 0, 0
    lh, ls = check_avl(node.left)
    rh, rs = check_avl(node.right)
    assert abs(lh - rh) <= 1, "AVL balance violated"
    assert node.height == max(lh, rh) + 1, "height annotation wrong"
    assert node.size == ls + rs + 1, "size annotation wrong"
    return node.height, node.size


def check_doc_invariants(node, depth=0):
    assert depth <= 32
    if isinstance(node, snapshots._Arr):
        _, size = check_avl(node.root)
        assert node.size == size
        for v in snapshots._tree_iter(node.root):
            check_doc_invariants(v, depth + 1)
    elif isinstance(node, snapshots._Obj):
        assert list(node.keys) == sorted(node.keys)
        assert len(node.keys) == len(node.vals) <= 32
        for v in node.vals:
            check_doc_invariants(v, depth + 1)


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

def test_prompt_example():
    s = Store([1, 2])
    v1 = s.apply(0, [{"op": "add", "path": ["0"], "value": 9}])
    assert v1 == 1
    assert s.read(1, []) == [9, 1, 2]
    assert s.read(0, []) == [1, 2]  # 版 0 不变


def test_root_ops():
    s = Store({"a": 1})
    v1 = s.apply(0, [{"op": "replace", "path": [], "value": [1, 2]}])
    assert s.read(v1, []) == [1, 2]
    v2 = s.apply(v1, [{"op": "add", "path": [], "value": {"b": 2}}])
    assert s.read(v2, []) == {"b": 2}
    expect_value_error(s.apply, v2, [{"op": "remove", "path": []}])  # 根不可删
    v3 = s.apply(0, [{"op": "test", "path": [], "value": {"a": 1}}])
    assert s.read(v3, []) == {"a": 1}
    s4 = Store(42)  # 标量根
    assert s4.read(0, []) == 42


def test_object_ops():
    s = Store({"a": 1, "b": 2})
    v1 = s.apply(0, [
        {"op": "add", "path": ["c"], "value": 3},   # 新字段
        {"op": "add", "path": ["a"], "value": 10},  # 覆盖字段
        {"op": "replace", "path": ["b"], "value": 20},
        {"op": "remove", "path": ["c"]},
    ])
    assert s.read(v1, []) == {"a": 10, "b": 20}
    expect_value_error(s.apply, 0, [{"op": "remove", "path": ["zz"]}])
    expect_value_error(s.apply, 0, [{"op": "replace", "path": ["zz"], "value": 1}])
    # 形似下标/'-' 的字段名按字段处理
    s2 = Store({"0": "zero", "-": "dash", "01": "o1"})
    assert s2.read(0, ["0"]) == "zero"
    assert s2.read(0, ["-"]) == "dash"
    assert s2.read(0, ["01"]) == "o1"


def test_array_ops():
    s = Store([1, 2, 3])
    v1 = s.apply(0, [
        {"op": "add", "path": ["0"], "value": 0},   # 头插 -> [0,1,2,3]
        {"op": "add", "path": ["-"], "value": 4},   # 追加 -> [0,1,2,3,4]
        {"op": "add", "path": ["5"], "value": 5},   # 下标==长度 -> 尾部
        {"op": "replace", "path": ["1"], "value": 10},
        {"op": "remove", "path": ["2"]},
    ])
    assert s.read(v1, []) == [0, 10, 3, 4, 5]
    expect_value_error(s.apply, 0, [{"op": "add", "path": ["4"], "value": 1}])  # 超长度
    expect_value_error(s.apply, 0, [{"op": "remove", "path": ["-"]}])
    expect_value_error(s.apply, 0, [{"op": "replace", "path": ["-"], "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "remove", "path": ["3"]}])
    for bad in ("01", "-1", "", "x", " 1", "1 ", "+1", "1.0", "１２"):
        expect_value_error(s.apply, 0,
                           [{"op": "replace", "path": [bad], "value": 1}])
        expect_value_error(s.read, 0, [bad])


def test_test_op():
    s = Store({"n": 1, "f": 1.0, "b": True, "l": [1, [2]], "s": "1", "z": None})
    s.apply(0, [{"op": "test", "path": ["n"], "value": 1.0}])  # 数字相等
    s.apply(0, [{"op": "test", "path": ["f"], "value": 1}])
    s.apply(0, [{"op": "test", "path": ["z"], "value": None}])
    s.apply(0, [{"op": "test", "path": ["l"], "value": [1, [2]]}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["b"], "value": 1}])   # 布尔≠数字
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["n"], "value": True}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["s"], "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["l"], "value": [1, [2, 3]]}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["l"], "value": [True, [2]]}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["zz"], "value": 1}])


def test_batch_sequential_and_empty():
    s = Store({})
    v1 = s.apply(0, [
        {"op": "add", "path": ["a"], "value": [1]},
        {"op": "add", "path": ["a", "-"], "value": 2},
        {"op": "test", "path": ["a"], "value": [1, 2]},  # 看到批内中间状态
        {"op": "replace", "path": ["a", "0"], "value": 9},
    ])
    assert s.read(v1, ["a"]) == [9, 2]
    v2 = s.apply(0, [])  # 空批也产生递增版本
    assert v2 == 2 and s.read(v2, []) == {}


def test_atomicity():
    s = Store({"a": [1, 2], "b": 1})
    before = s.version_count
    expect_value_error(s.apply, 0, [
        {"op": "replace", "path": ["b"], "value": 2},      # 成功
        {"op": "test", "path": ["b"], "value": 999},       # 失败 -> 整批作废
    ])
    assert s.version_count == before
    assert s.read(0, []) == {"a": [1, 2], "b": 1}
    expect_value_error(s.apply, 99, [])  # 非法版本也不新增
    assert s.version_count == before


def test_branching():
    s = Store([0])
    v1 = s.apply(0, [{"op": "add", "path": ["-"], "value": 1}])
    v2 = s.apply(0, [{"op": "add", "path": ["-"], "value": 2}])
    v3 = s.apply(v1, [{"op": "add", "path": ["-"], "value": 3}])
    assert (v1, v2, v3) == (1, 2, 3)
    assert s.read(v1, []) == [0, 1]
    assert s.read(v2, []) == [0, 2]
    assert s.read(v3, []) == [0, 1, 3]
    assert s.read(0, []) == [0]


def test_isolation():
    doc = {"a": [1, 2], "b": {"c": 3}}
    s = Store(doc)
    doc["a"].append(99)
    doc["b"]["c"] = 99
    doc["z"] = 1
    assert s.read(0, []) == {"a": [1, 2], "b": {"c": 3}}  # 改传入值不影响
    val = {"x": [1]}
    v1 = s.apply(0, [{"op": "add", "path": ["n"], "value": val}])
    val["x"].append(99)
    val["y"] = 2
    assert s.read(v1, ["n"]) == {"x": [1]}                # 改 op 的 value 不影响
    out = s.read(v1, [])
    out["n"]["x"].append(99)
    out["a"].append(99)
    assert s.read(v1, []) == {"a": [1, 2], "b": {"c": 3}, "n": {"x": [1]}}  # 改读取结果不影响


def test_limits():
    deep32 = "x"
    for _ in range(32):
        deep32 = [deep32]
    Store(deep32)                       # 深度 32 合法
    expect_value_error(Store, [deep32])  # 深度 33 非法
    s = Store(deep32)
    p31 = ["0"] * 31
    s.apply(0, [{"op": "replace", "path": p31, "value": ["x"]}])       # 31+1=32 合法
    expect_value_error(s.apply, 0,
                       [{"op": "replace", "path": p31, "value": [["x"]]}])  # 31+2=33 非法
    obj32 = {str(i): i for i in range(32)}
    Store(obj32)
    expect_value_error(Store, dict(obj32, **{"32": 32}))  # 33 字段非法
    s2 = Store(obj32)
    expect_value_error(s2.apply, 0, [{"op": "add", "path": ["zz"], "value": 1}])
    s2.apply(0, [{"op": "add", "path": ["0"], "value": 100}])  # 覆盖不增字段,合法
    big = Store([0] * 200000)
    expect_value_error(big.apply, 0, [{"op": "add", "path": ["-"], "value": 1}])
    expect_value_error(Store, [0] * 200001)


def test_invalid_inputs():
    s = Store([1])
    for bad in (-1, 1, 2, "0", 0.0, True, None, [0]):
        expect_value_error(s.apply, bad, [])
        expect_value_error(s.read, bad, [])
    expect_value_error(s.apply, 0, "not-a-list")
    expect_value_error(s.apply, 0, ["not-an-op"])
    expect_value_error(s.apply, 0, [{"op": "bogus", "path": []}])
    expect_value_error(s.apply, 0, [{"path": []}])
    expect_value_error(s.apply, 0, [{"op": "add", "path": []}])  # 缺 value
    expect_value_error(s.apply, 0, [{"op": "remove", "path": ["0"], "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "replace", "path": "0", "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "replace", "path": [0], "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "replace", "path": ["0"] * 33, "value": 1}])
    expect_value_error(s.apply, 0, [{"op": "test", "path": ["0"], "value": float("nan")}])
    expect_value_error(s.read, 0, "0")
    expect_value_error(s.read, 0, ["zz"])
    expect_value_error(s.read, 0, ["1"])
    expect_value_error(Store, object())
    expect_value_error(Store, {1: "x"})
    expect_value_error(Store, float("inf"))


def test_nested_paths():
    s = Store({"a": [{"b": [10, 20]}]})
    assert s.read(0, ["a", "0", "b", "1"]) == 20
    v1 = s.apply(0, [{"op": "replace", "path": ["a", "0", "b", "0"], "value": 99}])
    assert s.read(v1, []) == {"a": [{"b": [99, 20]}]}
    assert s.read(0, []) == {"a": [{"b": [10, 20]}]}


def test_structural_sharing():
    s = Store({"a": [1, 2, 3], "b": {"x": 1, "y": 2}})
    v1 = s.apply(0, [{"op": "replace", "path": ["b", "x"], "value": 9}])
    r0, r1 = s._versions[0], s._versions[1]
    assert r0 is not r1
    a0 = r0.vals[r0.keys.index("a")]
    a1 = r1.vals[r1.keys.index("a")]
    assert a0 is a1  # 未触碰的子树整体共享
    # 数组单点替换只复制脊柱,其余节点共享
    v2 = s.apply(0, [{"op": "replace", "path": ["a", "1"], "value": 99}])
    ar_old = s._versions[0].vals[0].root
    ar_new = s._versions[v2].vals[0].root
    old_ids = set()

    def collect(n):
        if n is not None:
            old_ids.add(id(n))
            collect(n.left)
            collect(n.right)

    collect(ar_old)
    shared = [0]

    def count(n):
        if n is not None:
            if id(n) in old_ids:
                shared[0] += 1
            count(n.left)
            count(n.right)

    count(ar_new)
    assert shared[0] >= 1, "no shared nodes after single replace"


def test_big_array_against_model():
    rng = random.Random(1234)
    model = []
    s = Store([])
    v = 0
    for i in range(4000):
        action = rng.random()
        if action < 0.4 or not model:
            pos = rng.randint(0, len(model))
            model.insert(pos, i)
            v = s.apply(v, [{"op": "add", "path": [str(pos)], "value": i}])
        elif action < 0.7:
            pos = rng.randrange(len(model))
            model[pos] = i
            v = s.apply(v, [{"op": "replace", "path": [str(pos)], "value": i}])
        else:
            pos = rng.randrange(len(model))
            del model[pos]
            v = s.apply(v, [{"op": "remove", "path": [str(pos)]}])
        if i % 997 == 0:
            assert s.read(v, []) == model
    assert s.read(v, []) == model
    check_doc_invariants(s._versions[v])


# ---------------------------------------------------------------------------
# 独立参照模型:深拷贝 + 普通 list/dict,与 snapshots 实现完全独立。
# ---------------------------------------------------------------------------

def ref_json_equal(a, b):
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(ref_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(ref_json_equal(a[k], b[k]) for k in a)
    return False


def ref_check_value(value, depth):
    if depth > 32:
        raise ValueError("too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite")
        return 0
    if depth == 32:
        raise ValueError("too deep")
    if isinstance(value, list):
        if len(value) > 200000:
            raise ValueError("array too long")
        return 1 + max((ref_check_value(v, depth + 1) for v in value), default=0)
    if isinstance(value, dict):
        if len(value) > 32:
            raise ValueError("object too wide")
        for k in value:
            if not isinstance(k, str):
                raise ValueError("bad key")
        return 1 + max((ref_check_value(v, depth + 1) for v in value.values()),
                       default=0)
    raise ValueError("bad type")


def ref_index(seg):
    if not seg or not seg.isascii() or not seg.isdigit():
        raise ValueError("bad index")
    if len(seg) > 1 and seg[0] == "0":
        raise ValueError("non-canonical index")
    return int(seg)


def ref_resolve(node, path):
    for seg in path:
        if isinstance(node, list):
            i = ref_index(seg)
            if i >= len(node):
                raise ValueError("index out of range")
            node = node[i]
        elif isinstance(node, dict):
            if seg not in node:
                raise ValueError("missing field")
            node = node[seg]
        else:
            raise ValueError("through scalar")
    return node


def ref_apply_op(doc, op):
    if not isinstance(op, dict):
        raise ValueError("bad op")
    kind = op.get("op")
    if kind not in ("add", "remove", "replace", "test"):
        raise ValueError("bad op kind")
    path = op.get("path")
    if (not isinstance(path, list)
            or any(not isinstance(s, str) for s in path)):
        raise ValueError("bad path")
    if len(path) > 32:
        raise ValueError("path too deep")
    if kind == "remove":
        if "value" in op:
            raise ValueError("remove has no value")
        if not path:
            raise ValueError("cannot remove root")
        parent = ref_resolve(doc, path[:-1])
        seg = path[-1]
        if isinstance(parent, dict):
            if seg not in parent:
                raise ValueError("missing field")
            del parent[seg]
        elif isinstance(parent, list):
            i = ref_index(seg)
            if i >= len(parent):
                raise ValueError("index out of range")
            del parent[i]
        else:
            raise ValueError("through scalar")
        return doc
    if "value" not in op:
        raise ValueError("missing value")
    value = op["value"]
    if kind == "test":
        target = ref_resolve(doc, path)
        ref_check_value(value, 0)
        if not ref_json_equal(target, value):
            raise ValueError("test failed")
        return doc
    ref_check_value(value, len(path))
    value = copy.deepcopy(value)
    if not path:
        return value
    parent = ref_resolve(doc, path[:-1])
    seg = path[-1]
    if isinstance(parent, dict):
        if kind == "add":
            if seg not in parent and len(parent) >= 32:
                raise ValueError("object too wide")
        elif seg not in parent:
            raise ValueError("missing field")
        parent[seg] = value
    elif isinstance(parent, list):
        if kind == "add":
            if seg == "-":
                i = len(parent)
            else:
                i = ref_index(seg)
                if i > len(parent):
                    raise ValueError("add beyond length")
            if len(parent) >= 200000:
                raise ValueError("array too long")
            parent.insert(i, value)
        else:
            i = ref_index(seg)
            if i >= len(parent):
                raise ValueError("index out of range")
            parent[i] = value
    else:
        raise ValueError("through scalar")
    return doc


def ref_apply(ref_versions, base, ops):
    if (isinstance(base, bool) or not isinstance(base, int)
            or not 0 <= base < len(ref_versions)):
        raise ValueError("bad version")
    if not isinstance(ops, list):
        raise ValueError("bad ops")
    doc = copy.deepcopy(ref_versions[base])
    for op in ops:
        doc = ref_apply_op(doc, op)
    ref_versions.append(doc)
    return len(ref_versions) - 1


# ---------------------------------------------------------------------------
# 差分模糊测试
# ---------------------------------------------------------------------------

FIELD_POOL = ["a", "b", "c", "d", "0", "1", "2", "-", "", "01", "x y"]
INDEX_POOL = ["0", "1", "2", "3", "4", "5", "6", "9", "-", "01", "-1", "",
              "x", " 1", "1 "]
SCALAR_POOL = [None, True, False, 0, 1, 2, -3, 1.5, 0.5, "", "0", "-", "a"]


def rand_value(rng, depth):
    if depth <= 0 or rng.random() < 0.55:
        return rng.choice(SCALAR_POOL)
    if rng.random() < 0.5:
        return [rand_value(rng, depth - 1) for _ in range(rng.randint(0, 3))]
    return {rng.choice(FIELD_POOL) + (str(i) if rng.random() < 0.3 else ""):
            rand_value(rng, depth - 1) for i in range(rng.randint(0, 3))}


def rand_path(rng, doc):
    path = []
    node = doc
    while rng.random() < 0.75:
        if isinstance(node, list) and node:
            if rng.random() < 0.85:
                i = rng.randrange(len(node))
                path.append(str(i))
                node = node[i]
            else:
                path.append(rng.choice(INDEX_POOL))
                return path
        elif isinstance(node, dict) and node:
            k = rng.choice(list(node.keys()) + [rng.choice(FIELD_POOL)])
            path.append(k)
            if k not in node:
                return path
            node = node[k]
        else:
            if rng.random() < 0.3:
                path.append(rng.choice(FIELD_POOL + INDEX_POOL))
            return path
    return path


def rand_op(rng, doc):
    kind = rng.choices(["add", "remove", "replace", "test"],
                       weights=[4, 2, 3, 2])[0]
    op = {"op": kind, "path": rand_path(rng, doc)}
    if kind != "remove":
        op["value"] = rand_value(rng, rng.choice([0, 1, 2, 3]))
    if rng.random() < 0.03:  # 少量畸形操作
        corrupt = rng.choice(["badop", "badpath", "dropvalue", "addvalue"])
        if corrupt == "badop":
            op["op"] = "bogus"
        elif corrupt == "badpath":
            op["path"] = rng.choice(["x", [0], [None], 5])
        elif corrupt == "dropvalue":
            op.pop("value", None)
        else:
            op["value"] = 1
    return op


def differential_fuzz():
    rng = random.Random(20261115)
    batches = 0
    for _ in range(60):
        doc = rand_value(rng, 3)
        store = Store(doc)
        ref_versions = [copy.deepcopy(doc)]
        for _ in range(25):
            base = len(ref_versions) - 1
            if rng.random() < 0.3:
                base = rng.randrange(len(ref_versions))
            if rng.random() < 0.05:
                base = rng.choice([-1, len(ref_versions), len(ref_versions) + 5,
                                   "0", True])
            valid_base = (isinstance(base, int) and not isinstance(base, bool)
                          and 0 <= base < len(ref_versions))
            walk_doc = ref_versions[base] if valid_base else ref_versions[-1]
            ops = [rand_op(rng, walk_doc) for _ in range(rng.randint(1, 6))]

            store_err = ref_err = None
            store_v = ref_v = None
            try:
                store_v = store.apply(base, ops)
            except ValueError as e:
                store_err = e
            try:
                ref_v = ref_apply(ref_versions, base, ops)
            except ValueError as e:
                ref_err = e

            assert (store_err is None) == (ref_err is None), (
                "divergence: store_err=%r ref_err=%r base=%r ops=%r"
                % (store_err, ref_err, base, ops))
            assert store.version_count == len(ref_versions), (
                "version count diverged: %d vs %d"
                % (store.version_count, len(ref_versions)))
            if store_err is not None:
                continue
            batches += 1
            assert store_v == ref_v == store.version_count - 1
            got = store.read(store_v, [])
            assert ref_json_equal(got, ref_versions[-1]), (
                "doc mismatch: %r vs %r" % (got, ref_versions[-1]))
            check_doc_invariants(store._versions[store_v])
            for _ in range(3):  # 随机路径读取比对(含报错行为)
                ver = rng.randrange(len(ref_versions))
                path = rand_path(rng, ref_versions[ver])
                s_ok = r_ok = False
                s_val = r_val = None
                try:
                    s_val = store.read(ver, path)
                    s_ok = True
                except ValueError:
                    pass
                try:
                    r_val = copy.deepcopy(ref_resolve(ref_versions[ver], path))
                    r_ok = True
                except ValueError:
                    pass
                assert s_ok == r_ok, ("read divergence", ver, path)
                if s_ok:
                    assert ref_json_equal(s_val, r_val), (
                        "read mismatch", ver, path, s_val, r_val)
            if batches % 8 == 0:  # 定期全量比对所有版本
                for ver in range(len(ref_versions)):
                    assert ref_json_equal(store.read(ver, []),
                                          ref_versions[ver])
    print("  差分模糊测试: %d 个成功批次全部与参照模型一致" % batches)


# ---------------------------------------------------------------------------
# 长数组头部插删资源实测
# ---------------------------------------------------------------------------

def _tree_node_ids(root):
    ids = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        ids.add(id(n))
        stack.append(n.left)
        stack.append(n.right)
    return ids


def benchmark():
    print("\n=== 长数组头部插删资源实测 ===")
    sizes = [10_000, 50_000, 200_000]
    k = 2000
    print("%10s %14s %14s %18s" % ("数组长度", "头插 µs/op", "头删 µs/op",
                                   "整表复制 µs/op"))
    for n in sizes:
        s = Store([0] * (n - k))  # 头插 k 次后正好达到长度 n
        v = 0
        t0 = time.perf_counter()
        for i in range(k):
            v = s.apply(v, [{"op": "add", "path": ["0"], "value": i}])
        ins_us = (time.perf_counter() - t0) / k * 1e6
        t0 = time.perf_counter()
        for _ in range(k):
            v = s.apply(v, [{"op": "remove", "path": ["0"]}])
        rem_us = (time.perf_counter() - t0) / k * 1e6
        assert s.read(v, []) == [0] * (n - k)  # k 插 + k 删后回到原内容
        lst = [0] * n
        r = 100
        t0 = time.perf_counter()
        for i in range(r):
            lst = [i] + lst  # 整数组复制的朴素做法,作对照
        naive_us = (time.perf_counter() - t0) / r * 1e6
        print("%10d %14.1f %14.1f %18.1f" % (n, ins_us, rem_us, naive_us))

    # 结构共享实测:一次头插的新增节点数(应为 O(log n) 量级)
    n = 200_000
    s = Store([0] * (n - 1))  # 头插 1 次后长度正好 n
    old_ids = _tree_node_ids(s._versions[0].root)
    v1 = s.apply(0, [{"op": "add", "path": ["0"], "value": -1}])
    new_root = s._versions[v1].root
    total = new = 0
    stack = [new_root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        total += 1
        if id(node) not in old_ids:
            new += 1
        stack.append(node.left)
        stack.append(node.right)
    print("头插 1 次(n=%d): 新增节点 %d 个 / 共 %d 个, 共享 %d 个 (log2(n)=%.1f)"
          % (n, new, total, total - new, math.log2(n)))
    assert new <= 3 * math.log2(n + 2) + 10, "新增节点数超出 O(log n) 量级"
    assert total - new >= n - 64, "共享节点过少"

    # 内存实测:500 次头插的增量(整表复制每次约需 n*8 字节)
    s2 = Store([0] * (n - 500))
    v = 0
    tracemalloc.start()
    for i in range(500):
        v = s2.apply(v, [{"op": "add", "path": ["0"], "value": i}])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("500 次头插(n=%d): 峰值内存 %.0f KiB ≈ %.1f KiB/次 "
          "(整表复制约需 %.0f KiB/次)"
          % (n, peak / 1024, peak / 500 / 1024, n * 8 / 1024))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("题目示例", test_prompt_example),
        ("根操作", test_root_ops),
        ("对象操作", test_object_ops),
        ("数组操作与下标规范", test_array_ops),
        ("test 操作与 JSON 相等", test_test_op),
        ("批内顺序执行与空批", test_batch_sequential_and_empty),
        ("批原子性", test_atomicity),
        ("多分支派生", test_branching),
        ("传入值/读取结果隔离", test_isolation),
        ("深度/字段数/长度上限", test_limits),
        ("非法输入", test_invalid_inputs),
        ("嵌套路径", test_nested_paths),
        ("结构共享(白盒)", test_structural_sharing),
        ("大数组随机操作对照", test_big_array_against_model),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   %s" % name)
        except Exception:
            failures += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    print("\n=== 独立参照核验(差分模糊测试) ===")
    try:
        differential_fuzz()
        print("ok   独立参照核验")
    except Exception:
        failures += 1
        print("FAIL 独立参照核验")
        traceback.print_exc()
    try:
        benchmark()
    except Exception:
        failures += 1
        print("FAIL 资源实测")
        traceback.print_exc()
    if failures:
        print("\n%d 项失败" % failures)
        return 1
    print("\n全部核验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
