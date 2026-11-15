"""snapshots 演示:题目示例、分支派生、批原子性、隔离性、共享/复杂度说明。"""

import snapshots


def show(title):
    print("\n=== %s ===" % title)


def main():
    show("题目示例:初值 [1,2],对版本 0 的 ['0'] add 9")
    store = snapshots.Store([1, 2])
    v1 = store.apply(0, [{"op": "add", "path": ["0"], "value": 9}])
    print("apply 返回新版本号:", v1)
    print("read(%d, []) =" % v1, store.read(v1, []))  # [9, 1, 2]
    print("read(0, []) =", store.read(0, []))        # [1, 2] 版 0 不变

    show("可从任一旧版本派生(分支)")
    v2 = store.apply(0, [{"op": "add", "path": ["-"], "value": 3}])
    print("从版本 0 再派生 -> 版本", v2, ":", store.read(v2, []))
    print("版本", v1, "仍为:", store.read(v1, []))

    show("test 操作:布尔与数字不同")
    s2 = snapshots.Store({"ok": True, "n": 1})
    try:
        s2.apply(0, [{"op": "test", "path": ["ok"], "value": 1}])
    except ValueError as e:
        print("test true==1 被拒绝:", e)
    v = s2.apply(0, [{"op": "test", "path": ["n"], "value": 1.0}])
    print("test 1==1.0(同为 JSON 数字)通过,版本", v)

    show("批原子性:任一操作失败,整批不新增版本")
    before = s2.version_count
    try:
        s2.apply(0, [
            {"op": "replace", "path": ["n"], "value": 42},  # 本可成功
            {"op": "test", "path": ["n"], "value": 999},    # 失败 -> 整批作废
        ])
    except ValueError as e:
        print("批内 test 失败:", e)
    print("版本数:", before, "->", s2.version_count, "(未新增)")
    print("版本 0 仍是:", s2.read(0, []))

    show("隔离性:修改传入值或读取结果不影响版本")
    doc = {"a": [1, 2]}
    s3 = snapshots.Store(doc)
    doc["a"].append(999)
    print("改动传入的 doc 后,read(0, []) =", s3.read(0, []))
    out = s3.read(0, ["a"])
    out.append(999)
    print("改动读取结果后,read(0, ['a']) =", s3.read(0, ["a"]))

    show("结构共享 / 原子性说明")
    print("数组 = 带子树大小的持久化 AVL 树:定位/插删/替换每层只复制")
    print("根到目标路径上的 O(log(n+2)) 个节点,其余子树按引用共享,")
    print("不做整数组扫描或复制;对象字段≤32,用有序元组,更新为常数级。")
    print("更新全是纯函数,版本表只追加:批内全部成功才登记新版本,")
    print("任何失败在追加前抛 ValueError,失败的批不产生版本。")
    print("\n运行 python3 verify.py 查看独立参照核验与长数组头部插删实测。")


if __name__ == "__main__":
    main()
