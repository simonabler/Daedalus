"""Fixture module with intentionally bad code to trigger all 8 smell rules.

Used exclusively by tests/test_smell_detector.py — not production code.

Expected smells:
- LongFunction      → very_long_function (> 50 lines)
- HighComplexity    → complex_function (> 10 branches)
- GodClass          → BigClass (> 20 methods)
- LongParameterList → many_params (> 5 params)
- DuplicateCode     → dup_alpha / dup_beta (≥ 80 % similar)
- DeadCode          → orphan_func (never called — requires call graph)
- MagicNumber       → uses_magic (literal 42 in function body)
- DeeplyNested      → deeply_nested_func (depth > 4)
"""


# ── Rule 1: Long Function (> 50 lines) ──────────────────────────────────────
def very_long_function(x):
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    i = h + 9
    j = i + 10
    k = j + 11
    l = k + 12
    m = l + 13
    n = m + 14
    o = n + 15
    p = o + 16
    q = p + 17
    r = q + 18
    s = r + 19
    t = s + 20
    u = t + 21
    v = u + 22
    w = v + 23
    x2 = w + 24
    y = x2 + 25
    z = y + 26
    aa = z + 27
    bb = aa + 28
    cc = bb + 29
    dd = cc + 30
    ee = dd + 31
    ff = ee + 32
    gg = ff + 33
    hh = gg + 34
    ii = hh + 35
    jj = ii + 36
    kk = jj + 37
    ll = kk + 38
    mm = ll + 39
    nn = mm + 40
    oo = nn + 41
    pp = oo + 42
    qq = pp + 43
    rr = qq + 44
    ss = rr + 45
    tt = ss + 46
    uu = tt + 47
    vv = uu + 48
    ww = vv + 49
    xx = ww + 50
    yy = xx + 51
    return yy


# ── Rule 2: High Complexity (> 10 branches) ─────────────────────────────────
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                result = a + b + c
            else:
                result = a + b
        elif d > 0:
            result = a + d
        else:
            result = a
    elif b > 0:
        if c > 0 and d > 0:
            result = b + c + d
        elif e > 0:
            result = b + e
        else:
            result = b
    else:
        result = 0
    for i in range(result):
        if i % 2 == 0 or i % 3 == 0:
            result += 1
    while result > 100:
        result -= 1
    return result


# ── Rule 3: God Class (> 20 methods) ────────────────────────────────────────
class BigClass:
    def method_01(self): pass
    def method_02(self): pass
    def method_03(self): pass
    def method_04(self): pass
    def method_05(self): pass
    def method_06(self): pass
    def method_07(self): pass
    def method_08(self): pass
    def method_09(self): pass
    def method_10(self): pass
    def method_11(self): pass
    def method_12(self): pass
    def method_13(self): pass
    def method_14(self): pass
    def method_15(self): pass
    def method_16(self): pass
    def method_17(self): pass
    def method_18(self): pass
    def method_19(self): pass
    def method_20(self): pass
    def method_21(self): pass  # exceeds limit of 20


# ── Rule 4: Long Parameter List (> 5 params) ─────────────────────────────────
def many_params(a, b, c, d, e, f, g):
    return a + b + c + d + e + f + g


# ── Rule 5: Duplicate Code (≥ 80 % similar bodies) ──────────────────────────
def dup_alpha(items):
    result = []
    for item in items:
        if item > 0:
            value = item * 2
            result.append(value)
        else:
            value = item * -1
            result.append(value)
    total = sum(result)
    return total


def dup_beta(elements):
    result = []
    for item in elements:
        if item > 0:
            value = item * 2
            result.append(value)
        else:
            value = item * -1
            result.append(value)
    total = sum(result)
    return total


# ── Rule 6: Dead Code ────────────────────────────────────────────────────────
def orphan_func():
    """This function is never called — requires call-graph data to detect."""
    return "I am dead code"


# ── Rule 7: Magic Numbers ────────────────────────────────────────────────────
def uses_magic(x):
    threshold = x * 42          # magic: 42
    offset = threshold + 999    # magic: 999
    return offset


# ── Rule 8: Deeply Nested (depth > 4) ───────────────────────────────────────
def deeply_nested_func(data):
    result = []
    for item in data:              # depth 1
        if item:                   # depth 2
            for sub in item:       # depth 3
                if sub:            # depth 4
                    for x in sub:  # depth 5 — exceeds limit
                        result.append(x)
    return result
