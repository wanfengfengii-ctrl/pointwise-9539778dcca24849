"""snapshots -- 持久化(不可变、多版本)JSON 文档快照库,纯 Python 标准库。

接口
----
``Store(document)`` 以 ``document`` 为版本 0 创建仓库;
``apply(base, ops)`` 从任一旧版本 ``base`` 派生,依次执行一批操作,
成功后返回递增的新版本号;``read(version, path)`` 返回该版本 ``path``
处的普通 JSON 值。

路径是字符串段列表,``[]`` 指根。对象段是字段名;数组段必须是规范非负
十进制下标("0"、"1"、……,不允许前导零、符号或空白)。数组 ``add``
允许下标等于数组长度(尾部插入)或 ``"-"``(追加)。

操作
----
ops 项为 ``{"op": ..., "path": [...], "value": ...}``:

* ``add``     父路径须存在;数组中按下标插入,对象中设置/覆盖字段;
              空路径表示整体替换根(根始终存在)。
* ``remove``  目标须存在;无 ``value``;根不可删除。
* ``replace`` 目标须存在;根可替换。
* ``test``    目标须存在且与 ``value`` JSON 相等;布尔与数字不同
              (``true != 1``),而 ``1`` 与 ``1.0`` 是相等的 JSON 数字。

结构共享
--------
数组用「带子树大小的 AVL 树」持久化实现,对象用「有序键元组」实现
(字段≤32,一次更新只复制一条 ≤32 项的元组,视为 O(1))。每次更新只
复制从根到被改位置路径上的节点,其余子树按引用共享,因此未改结构在
版本间共享;每层数组定位/插删/替换都是 O(log(n+2)),绝不做整数组
扫描或复制。初始化导入、``read`` 导出与 ``test`` 比较按所涉值大小
处理,这是契约允许的。

原子性
------
所有更新都是纯函数:输入当前根,返回新根,不就地修改任何东西。
``apply`` 把整批操作依次折叠在基版本的根上,只有全部成功后才把结果
追加进版本表;任何失败(非法版本/路径/操作、目标缺失、``test`` 失败)
都在追加之前抛出 ``ValueError``,因此失败的批不会新增版本,已有版本
全部不受影响。

隔离性
------
传入的值在导入时被深拷贝为全新的内部节点,``read`` 导出的是全新的
普通 list/dict,因此修改传入值或读取结果都不可能影响已存版本。

约束:文档深度 ≤ 32,对象 ≤ 32 字段,数组 ≤ 200000 项;违反即
``ValueError``。
"""

import bisect
import math

__all__ = ["Store"]

MAX_DEPTH = 32
MAX_OBJECT_FIELDS = 32
MAX_ARRAY_LEN = 200000

_MISSING = object()


# ---------------------------------------------------------------------------
# 持久化 AVL 树(数组)。节点不可变,更新沿路径复制。
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("left", "value", "right", "size", "height")

    def __init__(self, left, value, right):
        self.left = left
        self.value = value
        self.right = right
        self.size = _size(left) + 1 + _size(right)
        self.height = max(_height(left), _height(right)) + 1


def _size(node):
    return node.size if node is not None else 0


def _height(node):
    return node.height if node is not None else 0


def _mk(left, value, right):
    return _Node(left, value, right)


def _rot_left(node):
    r = node.right
    return _mk(_mk(node.left, node.value, r.left), r.value, r.right)


def _rot_right(node):
    l = node.left
    return _mk(l.left, l.value, _mk(l.right, node.value, node.right))


def _balanced(node):
    balance = _height(node.left) - _height(node.right)
    if balance > 1:
        left = node.left
        if _height(left.left) < _height(left.right):  # 左右情形,先左旋
            return _rot_right(_mk(_rot_left(left), node.value, node.right))
        return _rot_right(node)
    if balance < -1:
        right = node.right
        if _height(right.right) < _height(right.left):  # 右左情形,先右旋
            return _rot_left(_mk(node.left, node.value, _rot_right(right)))
        return _rot_left(node)
    return node


def _tree_get(node, index):
    while node is not None:
        left_size = _size(node.left)
        if index < left_size:
            node = node.left
        elif index > left_size:
            index -= left_size + 1
            node = node.right
        else:
            return node.value
    raise IndexError("tree index out of range")


def _tree_set(node, index, value):
    left_size = _size(node.left)
    if index < left_size:
        return _mk(_tree_set(node.left, index, value), node.value, node.right)
    if index > left_size:
        return _mk(node.left, node.value,
                   _tree_set(node.right, index - left_size - 1, value))
    return _mk(node.left, value, node.right)


def _tree_insert(node, index, value):
    if node is None:
        return _Node(None, value, None)
    left_size = _size(node.left)
    if index <= left_size:
        return _balanced(
            _mk(_tree_insert(node.left, index, value), node.value, node.right))
    return _balanced(
        _mk(node.left, node.value,
            _tree_insert(node.right, index - left_size - 1, value)))


def _tree_delete_max(node):
    if node.right is None:
        return node.left, node.value
    new_right, value = _tree_delete_max(node.right)
    return _balanced(_mk(node.left, node.value, new_right)), value


def _tree_delete(node, index):
    left_size = _size(node.left)
    if index < left_size:
        return _balanced(
            _mk(_tree_delete(node.left, index), node.value, node.right))
    if index > left_size:
        return _balanced(
            _mk(node.left, node.value,
                _tree_delete(node.right, index - left_size - 1)))
    if node.left is None:
        return node.right
    if node.right is None:
        return node.left
    new_left, value = _tree_delete_max(node.left)
    return _balanced(_mk(new_left, value, node.right))


def _tree_from_list(items):
    def build(lo, hi):
        if lo >= hi:
            return None
        mid = (lo + hi) >> 1
        return _Node(build(lo, mid), items[mid], build(mid + 1, hi))

    return build(0, len(items))


def _tree_iter(node):
    stack = []
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node.value
        node = node.right


# ---------------------------------------------------------------------------
# 内部容器:数组(_Arr,AVL 树)与对象(_Obj,有序键元组,≤32 字段)。
# ---------------------------------------------------------------------------

class _Arr:
    __slots__ = ("root", "size")

    def __init__(self, root):
        self.root = root
        self.size = _size(root)


class _Obj:
    __slots__ = ("keys", "vals")

    def __init__(self, keys, vals):
        self.keys = keys  # 升序 str 元组
        self.vals = vals  # 与 keys 平行的值元组


def _obj_index(obj, key):
    i = bisect.bisect_left(obj.keys, key)
    if i < len(obj.keys) and obj.keys[i] == key:
        return i
    return -1


def _obj_set(obj, key, value):
    i = bisect.bisect_left(obj.keys, key)
    if i < len(obj.keys) and obj.keys[i] == key:
        vals = list(obj.vals)
        vals[i] = value
        return _Obj(obj.keys, tuple(vals))
    return _Obj(obj.keys[:i] + (key,) + obj.keys[i:],
                obj.vals[:i] + (value,) + obj.vals[i:])


def _obj_del(obj, key):
    i = bisect.bisect_left(obj.keys, key)
    return _Obj(obj.keys[:i] + obj.keys[i + 1:],
                obj.vals[:i] + obj.vals[i + 1:])


# ---------------------------------------------------------------------------
# 导入 / 导出 / JSON 相等。
# ---------------------------------------------------------------------------

def _import(value, depth):
    """把普通 JSON 值深拷贝为内部持久形式。

    ``depth`` 是该值在文档中的位置深度(根为 0)。返回
    ``(节点, 值深度)``,值深度按容器层数计(标量为 0)。
    """
    if depth > MAX_DEPTH:
        raise ValueError("document deeper than 32 levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value, 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number is not valid JSON")
        return value, 0
    if depth == MAX_DEPTH:
        raise ValueError("document deeper than 32 levels")
    if isinstance(value, list):
        if len(value) > MAX_ARRAY_LEN:
            raise ValueError("array has more than 200000 items")
        nodes = []
        deepest = 0
        for item in value:
            node, vd = _import(item, depth + 1)
            nodes.append(node)
            if vd > deepest:
                deepest = vd
        return _Arr(_tree_from_list(nodes)), deepest + 1
    if isinstance(value, dict):
        if len(value) > MAX_OBJECT_FIELDS:
            raise ValueError("object has more than 32 fields")
        for key in value:
            if not isinstance(key, str):
                raise ValueError("object field names must be strings")
        keys = []
        nodes = []
        deepest = 0
        for key in sorted(value):
            node, vd = _import(value[key], depth + 1)
            keys.append(key)
            nodes.append(node)
            if vd > deepest:
                deepest = vd
        return _Obj(tuple(keys), tuple(nodes)), deepest + 1
    raise ValueError("unsupported JSON value type: %s" % type(value).__name__)


def _export(node):
    """把内部形式导出为全新的普通 JSON 值(调用方改动不影响仓库)。"""
    if isinstance(node, _Arr):
        return [_export(item) for item in _tree_iter(node.root)]
    if isinstance(node, _Obj):
        return {key: _export(val) for key, val in zip(node.keys, node.vals)}
    return node


def _json_equal(a, b):
    """JSON 相等:布尔与数字不同;1 与 1.0 作为数字相等。"""
    if a is b:
        return True
    if isinstance(a, _Arr):
        return (isinstance(b, _Arr) and a.size == b.size and
                all(_json_equal(x, y)
                    for x, y in zip(_tree_iter(a.root), _tree_iter(b.root))))
    if isinstance(a, _Obj):
        return (isinstance(b, _Obj) and a.keys == b.keys and
                all(_json_equal(x, y) for x, y in zip(a.vals, b.vals)))
    if isinstance(a, bool) or isinstance(b, bool):
        return False  # 布尔是单例;a is b 已失败,说明一个是布尔另一个不是同一个
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False  # None 与非 None,或类型不匹配


# ---------------------------------------------------------------------------
# 路径解析与持久化修改。
# ---------------------------------------------------------------------------

def _parse_index(seg):
    if not seg or not seg.isascii() or not seg.isdigit():
        raise ValueError("invalid array index segment: %r" % (seg,))
    if len(seg) > 1 and seg[0] == "0":
        raise ValueError("non-canonical array index segment: %r" % (seg,))
    return int(seg)


def _child(node, seg):
    if isinstance(node, _Arr):
        index = _parse_index(seg)
        if index >= node.size:
            raise ValueError("array index out of range: %r" % (seg,))
        return _tree_get(node.root, index)
    if isinstance(node, _Obj):
        i = _obj_index(node, seg)
        if i < 0:
            raise ValueError("missing object field: %r" % (seg,))
        return node.vals[i]
    raise ValueError("path descends into a scalar at segment %r" % (seg,))


def _resolve(root, path):
    node = root
    for seg in path:
        node = _child(node, seg)
    return node


def _replace_child(node, seg, new_child):
    # node 是容器且 seg 指向已存在的子节点(调用前已由 _child 验证)。
    if isinstance(node, _Arr):
        return _Arr(_tree_set(node.root, _parse_index(seg), new_child))
    return _obj_set(node, seg, new_child)


def _modify(node, path, pos, op, value):
    if pos == len(path):
        # 目标就是当前节点本身(只在空路径即根处发生):add/replace 整体替换。
        if op == "remove":
            raise ValueError("the root cannot be removed")
        return value
    seg = path[pos]
    if pos == len(path) - 1:
        return _modify_leaf(node, seg, op, value)
    child = _child(node, seg)  # 验证中间段存在且合法
    return _replace_child(node, seg, _modify(child, path, pos + 1, op, value))


def _modify_leaf(node, seg, op, value):
    if isinstance(node, _Obj):
        i = _obj_index(node, seg)
        if op == "add":
            if i < 0 and len(node.keys) >= MAX_OBJECT_FIELDS:
                raise ValueError("object would exceed 32 fields")
            return _obj_set(node, seg, value)
        if i < 0:
            raise ValueError("missing object field: %r" % (seg,))
        if op == "remove":
            return _obj_del(node, seg)
        return _obj_set(node, seg, value)
    if isinstance(node, _Arr):
        if op == "add":
            if seg == "-":
                index = node.size
            else:
                index = _parse_index(seg)
                if index > node.size:
                    raise ValueError(
                        "add index beyond array length: %r" % (seg,))
            if node.size >= MAX_ARRAY_LEN:
                raise ValueError("array would exceed 200000 items")
            return _Arr(_tree_insert(node.root, index, value))
        index = _parse_index(seg)
        if index >= node.size:
            raise ValueError("array index out of range: %r" % (seg,))
        if op == "remove":
            return _Arr(_tree_delete(node.root, index))
        return _Arr(_tree_set(node.root, index, value))
    raise ValueError("path descends into a scalar at segment %r" % (seg,))


# ---------------------------------------------------------------------------
# 仓库。
# ---------------------------------------------------------------------------

class Store:
    """版本化、持久化的 JSON 文档仓库。语义详见模块文档字符串。"""

    def __init__(self, document):
        root, _ = _import(document, 0)
        self._versions = [root]  # 下标即版本号,只追加

    @property
    def version_count(self):
        return len(self._versions)

    def _root(self, version):
        if (isinstance(version, bool) or not isinstance(version, int)
                or version < 0 or version >= len(self._versions)):
            raise ValueError("unknown version: %r" % (version,))
        return self._versions[version]

    def apply(self, base, ops):
        """在版本 ``base`` 上依次执行 ``ops``,全部成功则登记并返回
        递增的新版本号;任一操作非法或 test 失败则抛 ValueError,
        整批不新增版本。"""
        root = self._root(base)
        if (isinstance(ops, (str, bytes))
                or not isinstance(ops, (list, tuple))):
            raise ValueError("ops must be a list of operation objects")
        current = root
        for op in ops:
            current = self._apply_one(current, op)
        self._versions.append(current)  # 只有全部成功才会执行到这里
        return len(self._versions) - 1

    @staticmethod
    def _apply_one(root, op):
        if not isinstance(op, dict):
            raise ValueError("operation must be an object")
        kind = op.get("op")
        if kind not in ("add", "remove", "replace", "test"):
            raise ValueError("unknown operation: %r" % (kind,))
        path = op.get("path", _MISSING)
        if (path is _MISSING or isinstance(path, (str, bytes))
                or not isinstance(path, (list, tuple))
                or any(not isinstance(seg, str) for seg in path)):
            raise ValueError("operation path must be a list of strings")
        path = list(path)
        if len(path) > MAX_DEPTH:
            raise ValueError("path deeper than 32 segments")
        if kind == "remove":
            if "value" in op:
                raise ValueError("remove takes no value")
            if not path:
                raise ValueError("the root cannot be removed")
            return _modify(root, path, 0, "remove", None)
        if "value" not in op:
            raise ValueError("%s requires a value" % kind)
        if kind == "test":
            target = _resolve(root, path)
            expected, _ = _import(op["value"], 0)
            if not _json_equal(target, expected):
                raise ValueError("test operation failed")
            return root
        imported, _ = _import(op["value"], len(path))
        return _modify(root, path, 0, kind, imported)

    def read(self, version, path):
        """返回 ``version`` 中 ``path`` 处的普通 JSON 值(全新对象)。"""
        root = self._root(version)
        if (isinstance(path, (str, bytes))
                or not isinstance(path, (list, tuple))
                or any(not isinstance(seg, str) for seg in path)):
            raise ValueError("path must be a list of strings")
        return _export(_resolve(root, list(path)))
