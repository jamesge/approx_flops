from dataclasses import dataclass

try:
    import graphviz
except ModuleNotFoundError:
    graphviz = None

@dataclass
class VarMeta:
    value: float
    reductions: list[str]

g_var_registry = dict[str, VarMeta]()
g_merge_registry = dict[str, VarMeta]()

def eval_variable(var_name:str):
    mul = True
    if var_name.startswith('/'):
        mul = False
        var_name = var_name[1:]
    if not var_name in g_var_registry:
        raise Exception(f"Unknown variable={var_name}, set it with register_variable")
    val = g_var_registry[var_name].value
    return val if mul else 1/val

def parse_shape_names(shapedesc: str):
    return [x.strip() for x in shapedesc.split(',')]

def is_num(s):
    try: float(s)
    except ValueError: return False
    else: return True

def parse_atomic_symbols_from_list(exprlist: list[str]) -> tuple[float, list[str]]:
    const_factor = 1
    mul_names = []
    div_names = []
    for mul_name in exprlist:
        mul_name = mul_name.strip()
        if '/' in mul_name:
            more_splits = mul_name.split('/')
            mul_name = more_splits[0].strip()
            # could divide more than one: "D/T/H"
            for div_name in more_splits[1:]:
                if is_num(div_name): # constant, e.g. the "4" in "4*C"
                    const_factor /= float(div_name)
                else:
                    div_names.append(div_name)
        if is_num(mul_name): # constant, e.g. the "4" in "4*C"
            const_factor *= float(mul_name)
        else:
            mul_names.append(mul_name)
    
    for div_name in div_names:
        if div_name in mul_names:
            # cancel the multiplication
            del mul_names[mul_names.index(div_name)]
        else:
            # div_names are always started with '/' which is used in eval_variable
            mul_names.append('/' + div_name)
    return (const_factor, mul_names)

def parse_atomic_symbols(expr: str) -> tuple[float, list[str]]:
    return parse_atomic_symbols_from_list(expr.split('*'))
    
def get_varmeta_from_expr(expr: str):
    cf, mul_names = parse_atomic_symbols(expr)
    p = cf
    for name in mul_names:
        p *= eval_variable(name)
    reductions = [str(cf)]
    reductions.extend(mul_names)
    return VarMeta(p, reductions)

def get_int_from_expr(expr:str) -> int:
    cf, mul_names = parse_atomic_symbols(expr)
    p = cf
    for name in mul_names:
        p *= eval_variable(name)
    assert int(p) == p, f"Expect {p} to be integer"
    return int(p)

def simplify_expr(expr:str) -> str:
    cf, mul_names = parse_atomic_symbols(expr)
    return make_product_str(cf, mul_names)

def make_product_str(factor, names):
    l = []
    if factor != 1:
        l.append(f"{factor}")
    last_name_single_alphabet = True
    for name in names:
        if len(l) > 0 and (not last_name_single_alphabet) and not name.startswith('/'):
            l.append(f"*{name}")
        else:
            l.append(name)
        last_name_single_alphabet = (len(name) == 1)
    return ''.join(l)

# NOTE: could be overwritten
def register_variable(var_name:str, expr:int|float|str):
    if isinstance(expr, int) or isinstance(expr, float):
        g_var_registry[var_name] = VarMeta(expr, [])
    elif isinstance(expr, str):
        # handy to define variable based on product of other variables
        if not '*' in var_name:
            g_var_registry[var_name] = get_varmeta_from_expr(expr)
        else:
            g_merge_registry[var_name] = get_varmeta_from_expr(expr)
    else:
        raise Exception("Unsupported type of val")

def clear_variables():
    g_var_registry.clear()
    g_merge_registry.clear()
    
def get_shape_from_names(shape_names: list[str]):
    return [get_int_from_expr(var_expr) for var_expr in shape_names]

def numel_impl(shape:list):
    p:int = 1
    for sz in shape:
        p *= sz
    return p


class SymbolicProduct:
    def __init__(self, names:list[str], *, modifiable_input=False):
        self.names = []
        self.const_factor = 1

        if not modifiable_input:
            names = names[:]

        idx_to_be_removed = []

        self.const_factor, names = parse_atomic_symbols_from_list(names)

        # single-variable reduction
        # NOTE: can't use enumerate since names may be extended during iteration
        name_count = len(names)
        for i, n in enumerate(names):
            if n in g_var_registry and len(g_var_registry[n].reductions) > 0:
                idx_to_be_removed.append(i)
                names.extend(g_var_registry[n].reductions)
        for i in idx_to_be_removed[::-1]: del names[i]
        idx_to_be_removed.clear()

        # multi-variable reduction
        name_count = len(names)
        for i in range(name_count):
            for j in range(i+1, name_count):
                n_s = names[i]
                n_m = names[j]
                if n_s > n_m:
                    n_s, n_m = n_m, n_s
                n2 = f"{n_s}*{n_m}"
                if n2 in g_merge_registry and len(g_merge_registry[n2].reductions) > 0:
                    #print(f"found n2={n2} i={i} j={j} names={names}")
                    idx_to_be_removed.append(i)
                    idx_to_be_removed.append(j)
                    e = g_merge_registry[n2]
                    names[i] = "" # must be erased
                    names[j] = "" 
                    names.extend(e.reductions)
                    break
        for i in idx_to_be_removed[::-1]: del names[i]
        idx_to_be_removed.clear();
            
        for n in names:
            if is_num(n):
                self.const_factor *= float(n)
            else:
                for sub_var_name in n.split('*'):
                    sub_var_name = sub_var_name.strip()
                    if is_num(sub_var_name):
                        self.const_factor *= float(sub_var_name)
                    else:
                        self.names.append(sub_var_name)

        self.names.sort()

        # Since asiic code of '/' is less than alphabets, symbols to be divided
        # like '/X' will be sorted at the beginning, making expressions like
        # "/TP*BTCC" which is not natural to human, and there's no easy way
        # to change the sorting behavior.
        # As as workaround, we just move all symbols started with '/' to the end
        # of the array after sorting
        slash_count = 0
        while slash_count < len(self.names):
            if self.names[slash_count].startswith('/'):
                slash_count += 1
            else:
                break
        if slash_count > 0:
            self.names = self.names[slash_count:] + self.names[:slash_count]

    def __repr__(self):
        return make_product_str(self.const_factor, self.names)

    def equal(self, other) -> bool:
        if len(self.names) != len(other.names):
            return False
        for i, n in enumerate(self.names):
            if n != other.names[i]:
                return False
        return True

class SymbolicExpr:
    def __init__(self):
        self.sum_of_prods: list[SymbolicProduct] = []

    def add_product(self, new_prod:SymbolicProduct):
        for prod in self.sum_of_prods:
            if prod.equal(new_prod):
                # merge with existing product
                prod.const_factor += new_prod.const_factor
                return 
        self.sum_of_prods.append(new_prod)

    def __repr__(self):
        if len(self.sum_of_prods) == 0:
            return '<None>'
        return '+'.join([str(x) for x in self.sum_of_prods])

    def clear(self):
        self.sum_of_prods.clear()

# TODO: add range
class DeviceSpan:
    # only 1D device mesh right now
    def __init__(self, device_count_expr:str):
        self.device_count = get_int_from_expr(device_count_expr)
        self.device_count_expr = device_count_expr
        assert self.device_count > 1, "at least 2 devices"

    def equal(self, other):
        return self.device_count == other.device_count

    def __repr__(self):
        return f"Span({self.device_count_expr}={self.device_count})"

    @staticmethod
    def check_compat(dr1, dr2) -> bool:
        if (dr1 is None) or (dr2 is None):
            return (dr1 is None) and (dr2 is None)
        return dr1.device_count == dr2.device_count

class Stats:
    accum_flops = 0
    accum_flops_expr = SymbolicExpr()
    accum_comms = 0
    accum_comms_expr = SymbolicExpr()
    
    @staticmethod
    def flops_str():
        return f"{Stats.accum_flops_expr}={Stats.accum_flops}"

    @staticmethod
    def comms_str():
        return f"{Stats.accum_comms_expr}={Stats.accum_comms}"

    @staticmethod
    def add_flops(new_flops:int, new_flops_expr:SymbolicProduct, prefix:str):
        Stats.accum_flops += new_flops
        Stats.accum_flops_expr.add_product(new_flops_expr)
        print(f"{prefix} new_flops={new_flops_expr} total_flops={Stats.flops_str()}")

    @staticmethod
    def print_stats():
        print(f"TotalFlops: {Stats.flops_str()}")
        print(f"TotalComms: {Stats.comms_str()}")

    @staticmethod
    def clear_stats():
        Stats.accum_flops = 0
        Stats.accum_flops_expr.clear()
        Stats.accum_comms = 0
        Stats.accum_comms_expr.clear()

class GraphNode:
    node_id_alloc = 0
    def __init__(self, op_name:str, nodes):
        self.op_name = op_name
        self.children = []
        self.leaf = None
        self.node_id = GraphNode.node_id_alloc
        GraphNode.node_id_alloc += 1
        if isinstance(nodes, FakeTensor):
            self.leaf = nodes
        elif isinstance(nodes, GraphNode):
            self.children.append(nodes)
        elif isinstance(nodes, list):
           for node in nodes:
                if isinstance(node, GraphNode):
                    self.children.append(node)
                elif isinstance(node, FakeTensor):
                    self.children.append(GraphNode("wrap", node))
                else:
                    raise Exception(f"Unknown node type={node}")
        else:
            raise Exception("Unknown nodes types={nodes}")

    def __repr__(self):
        node_dedup = set()
        node_stack = []
        desc = "digraph G {\n"
        node_stack.append(self)
        while len(node_stack) > 0:
            node = node_stack.pop()
            if node.leaf is not None:
                desc += f"  {node.node_id} [label=\"{str(node.leaf)}\"];\n"
            else:
                desc += f"  {node.node_id} [label=\"{node.op_name}\"];\n"
                for child_node in node.children:
                    desc += f"  {child_node.node_id} -> {node.node_id};\n"
                    if not child_node.node_id in node_dedup:
                        node_dedup.add(child_node.node_id)
                        node_stack.append(child_node)
        desc += "}"
        return desc

def shape_to_str(shapelist):
    return f"({','.join(shapelist)})"
        
class FakeTensor:
    def __init__(self, shapedesc: str|None, device: DeviceSpan|None = None):
        self.shape = []
        if shapedesc != None:
            self.shape_names = parse_shape_names(shapedesc)
            self.shape = get_shape_from_names(self.shape_names)
            assert len(self.shape) == len(self.shape_names)
        else:
            self.shape_names = []
        self.device = device
        self.compute_graph: GraphNode|None = None

    def derive(self):
        r = FakeTensor(None)
        r.device = self.device
        r.compute_graph = self.compute_graph
        return r

    def clone(self):
        r = FakeTensor(None)
        r.shape_names = self.shape_names[:]
        r.shape = self.shape[:]
        r.device = self.device
        r.compute_graph = self.compute_graph
        return r

    def peek(self):
        print(f"Peek: {self}")
        return self

    def get_compute_graph(self):
        return self.compute_graph if self.compute_graph is not None else GraphNode("wrap", self)

    def show_compute_graph(self):
        assert graphviz is not None, "graphviz is required"
        print("Showing DAG...(may not popup)")
        root = self.get_compute_graph()
        graphviz.Source(str(root), filename=f"root_{root.node_id}.gv", format="png").view()

    def __repr__(self):
        desc = f"{shape_to_str(self.shape_names)}-{shape_to_str([str(x) for x in self.shape])}"
        if self.device is not None:
            desc += f"-{self.device}"
        return desc

    def axis_count(self):
        return len(self.shape)

    def transpose(self, axis1, axis2):
        if axis1 != axis2:
            r = self.clone()
            r.shape[axis1], r.shape[axis2] = self.shape[axis2], self.shape[axis1]
            r.shape_names[axis1], r.shape_names[axis2] = self.shape_names[axis2], self.shape_names[axis1]
            r.compute_graph = GraphNode(f"transpose({axis1},{axis2})", self.get_compute_graph())
            return r
        else:
            return self

    def squeeze_(self, dim: int|None = None):
        if dim is not None:
            if self.shape[dim] == 1:
                del self.shape[dim]
                del self.shape_names[dim]
            else:
                raise Exception(f"Expect dim_{dim} to be 1, actually {self.shape[dim]}")
        else:
            new_shape = []
            for i in range(len(self.shape)):
                if self.shape[i] == 1:
                    del self.shape_names[i]
                else:
                    new_shape.append(self.shape[i])
            self.shape = new_shape
        return self

    def unsqueeze_(self, dim:int):
        self.shape.insert(dim, 1)
        self.shape_names.insert(dim, "1")
        return self

    def repeat(self, repeat_count:str, dim:int):
        r = self.clone()
        r.shape.insert(dim, get_int_from_expr(repeat_count))
        r.shape_names.insert(dim, repeat_count)
        r.compute_graph = GraphNode(f"repeat({dim})->{shape_to_str(r.shape_names)}", self.get_compute_graph())
        return r

    def sum(self, dim:int, keepdim=False):
        r = self.clone()
        if keepdim:
            r.shape[dim] = 1
            r.shape_names[dim] = "1"
        else:
            del r.shape[dim]
            del r.shape_names[dim]
        r.compute_graph = GraphNode(f"sum({dim})->{r}", self.get_compute_graph())
        # not add these little flops right now
        #Stats.add_flops(self.numel(), SymbolicProduct(self.shape_names), f"Sum: {self} -> {r}")
        return r

    def layer_norm(self):
        r = self.clone()
        r.compute_graph = GraphNode(f"LN", self.get_compute_graph())
        # not add these little flops right now
        #Stats.add_flops(r.numel(), SymbolicProduct(r.shape_names), f"LN:")
        return r

    def softmax(self):
        r = self.clone()
        r.compute_graph = GraphNode(f"softmax", self.get_compute_graph())
        # not add these little flops right now
        #Stats.add_flops(r.numel(), SymbolicProduct(r.shape_names), f"Softmax:")
        return r        
            
    def view(self, newshape_desc):
        newshape_names = parse_shape_names(newshape_desc)
        newshape = get_shape_from_names(newshape_names)
        assert len(newshape) != 0 and len(self.shape) != 0
        i1 = i2 = 0
        p1 = p2 = 0
        while i1 < len(newshape) and i2 < len(self.shape):
            p1 = newshape[i1]
            p2 = self.shape[i2]
            if p1 < p2:
                while p1 < p2 and i1 < len(newshape)-1:
                    i1 += 1
                    p1 *= newshape[i1]
            elif p2 < p1:
                while p2 < p1 and i2 < len(self.shape)-1:
                    i2 += 1
                    p2 *= self.shape[i2]
            if p1 != p2:
                raise Exception(f"Cannot view {self.shape_names}-{self.shape} as ({newshape_desc})-{newshape}")
            i1 += 1
            i2 += 1
        while i1 < len(newshape):
            p1 *= self.shape[i1]
            i1 += 1
        while i2 < len(self.shape):
            p2 *= self.shape[i2]
            i2 += 1
        if p1 != p2:
            raise Exception(f"Cannot view {self.shape_names}-{self.shape} as ({newshape_desc})-{newshape}")
        r = self.derive()
        r.shape_names = newshape_names
        r.shape = newshape
        r.compute_graph = GraphNode(f"view({newshape_desc})", self.get_compute_graph())
        return r
    
    # if `causal' is True, the calculated flops will be half-ed
    def matmul(self, other, causal = False):
        if not isinstance(other, FakeTensor):
            raise Exception("ERR: other must be FakeTensor as well")
        assert other.axis_count() >= 2, "RHS tensor must have at least 2 dims"
        assert self.axis_count() >= 2, "LHS tensor must have at least 2 dims"
        assert self.shape[-1] == other.shape[-2], f"dims for matmul does not match, ({self.shape[-2]},{self.shape[-1]})@({other.shape[-2]},{other.shape[-1]}) self={self} other={other}"

        if not DeviceSpan.check_compat(self.device, other.device):
            raise Exception(f"self.device={self.device} is not compatible with other.device={other.device}")

        min_axis = min(other.axis_count(), self.axis_count())
        # check broadcastable dims
        for axis in range(2, min_axis):
            sz1 = self.shape[-axis-1]
            sz2 = other.shape[-axis-1]
            if sz1 != sz2 and sz1 != 1 and sz2 != 1:
                raise Exception(f"Unmatched dim at [{-axis}], one of them must be 1 to follow broadcasting rules, self={self} other={other}")
        max_axis = max(other.axis_count(), self.axis_count())
        new_shape = [0] * max_axis
        def safe_get(l, idx, def_val):
            return l[idx] if idx < len(l) and idx >= 0 else def_val
        ret = self.derive()
        ret.shape_names = ["-"] * len(new_shape)
        pad_self = max_axis - len(self.shape)
        pad_other = max_axis - len(other.shape)
        batch_factor = 1
        new_flops_prod_list = []
        for axis in range(0, max_axis - 2):
            d1 = safe_get(self.shape, axis - pad_self, 0)
            d2 = safe_get(other.shape, axis - pad_other, 0)
            new_d = max(d1, d2)
            new_name = self.shape_names[axis - pad_self] if d1 > d2 else other.shape_names[axis - pad_other]
            new_shape[axis] = new_d
            ret.shape_names[axis] = new_name
            batch_factor *= new_d
            new_flops_prod_list.append(new_name)
        new_shape[-2] = self.shape[-2]
        new_shape[-1] = other.shape[-1]
        ret.shape_names[-2] = self.shape_names[-2]
        ret.shape_names[-1] = other.shape_names[-1]
        ret.shape = new_shape
        ret.compute_graph = GraphNode(f"matmul->{shape_to_str(ret.shape_names)}", [self.get_compute_graph(), other.get_compute_graph()])
        new_flops = batch_factor * self.shape[-2] * self.shape[-1] * other.shape[-1]
        new_flops_prod_list.extend([self.shape_names[-2], self.shape_names[-1], other.shape_names[-1]])
        new_flops_expr = SymbolicProduct(new_flops_prod_list, modifiable_input=True)
        if not causal:
            new_flops *= 2
            new_flops_expr.const_factor *= 2
        Stats.add_flops(new_flops, new_flops_expr, f"Matmul: {self} @ {other} -> {ret}")
        return ret

    def numel(self):
        return numel_impl(self.shape)

    def all_gather(self, *, axis:int):
        assert self.device, "The tensor to collective ops must be created with a device"
        r = self.clone()
        r.shape[axis] *= self.device.device_count
        r.shape_names[axis] = simplify_expr(f"{r.shape_names[axis]}*{self.device.device_count_expr}")
        r.compute_graph = GraphNode(f"AG(axis={axis})", self.get_compute_graph())
        # update stats
        new_comms = r.numel()
        new_comms_expr = SymbolicProduct(r.shape_names)
        Stats.accum_comms += new_comms
        Stats.accum_comms_expr.add_product(new_comms_expr)
        print(f"AllGather: {self} -> {r}, new_comms={new_comms_expr} total_comms={Stats.comms_str()}")
        return r

    def reduce_scatter(self, *, scatter_axis:int):
        assert self.device, "The tensor to collective ops must be created with a device"
        r = self.clone()
        sz = r.shape[scatter_axis]
        assert sz % self.device.device_count == 0, f"shape[{scatter_axis}]={sz} is not a multiple of num_devices={self.device.device_count}"
        r.shape[scatter_axis] //= self.device.device_count
        r.shape_names[scatter_axis] = simplify_expr(f"{r.shape_names[scatter_axis]}/{self.device.device_count_expr}")
        r.compute_graph = GraphNode(f"RS(scatter={scatter_axis})", self.get_compute_graph())
        # update stats
        new_comms = self.numel()
        new_comms_expr = SymbolicProduct(self.shape_names)
        Stats.accum_comms += new_comms
        Stats.accum_comms_expr.add_product(new_comms_expr)
        print(f"ReduceScatter: {self} -> {r}, new_comms={new_comms_expr} total_comms={Stats.comms_str()}")
        return r

    def all_reduce(self):
        assert self.device, "The tensor to collective ops must be created with a device"
        r = self.clone()
        r.compute_graph = GraphNode(f"AR", self.get_compute_graph())
        # update stats
        new_comms = self.numel()
        new_comms_expr = SymbolicProduct(self.shape_names)
        Stats.accum_comms += new_comms
        Stats.accum_comms_expr.add_product(new_comms_expr)
        print(f"AllReduce: {self} -> {r}, new_comms={new_comms_expr} total_comms={Stats.comms_str()}")
        return r

    def all2all(self, *, gather_axis:int, scatter_axis:int):
        assert self.device, "The tensor to collective ops must be created with a device"
        r = self.clone()
        r.shape[gather_axis] *= self.device.device_count
        r.shape_names[gather_axis] = simplify_expr(f"{r.shape_names[gather_axis]}*{self.device.device_count_expr}")
        r.compute_graph = GraphNode(f"A2A(gather={gather_axis} scatter={scatter_axis})", self.get_compute_graph())

        sz = r.shape[scatter_axis]
        assert sz % self.device.device_count == 0, f"shape[{scatter_axis}]={sz} is not a multiple of num_devices={self.device.device_count}"
        r.shape[scatter_axis] //= self.device.device_count
        r.shape_names[scatter_axis] = simplify_expr(f"{r.shape_names[scatter_axis]}/{self.device.device_count_expr}")

        # update stats
        new_comms = self.numel()
        new_comms_expr = SymbolicProduct(self.shape_names)
        Stats.accum_comms += new_comms
        Stats.accum_comms_expr.add_product(new_comms_expr)
        print(f"All2All: {self} -> {r}, new_comms={new_comms_expr} total_comms={Stats.comms_str()}")        
        return r

def test_graph():
    register_variable("B", 2)
    register_variable("T", 4)
    register_variable("C", 6)    
    x = FakeTensor("B,T,C")
    y = FakeTensor("B,T,C")
    n1 = GraphNode("add", [x, y])
    n2 = GraphNode("mul", [n1, y])
    print(n2)

def single_device_example():
    dense = False
    
    register_variable("B", 2)
    register_variable("T", 8)
    register_variable("H", 2)
    register_variable("D", 4)
    register_variable("C", "D*H")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "C")
    register_variable("E", 16)
    register_variable("topK", 4)
    register_variable("Ce", "C")

    #print(f"g_var_registry: {g_var_registry}\ng_merge_registry: {g_merge_registry}");

    x = FakeTensor("B,T,C").layer_norm()
    all_x = FakeTensor("B,T,C").layer_norm()
    Wq = FakeTensor("C,C")
    Wk = FakeTensor("C,C")
    Wv = FakeTensor("C,C")
    q = x.matmul(Wq).view("B,T,H,D").transpose(1,2)
    k = all_x.matmul(Wk).view("B,T,H,D").transpose(1,2)
    v = all_x.matmul(Wv).view("B,T,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True).softmax()
    merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,T,C")
    Wo = FakeTensor("C,C")
    x2 = merged_v.matmul(Wo).layer_norm()

    if dense:
        fc1 = FakeTensor("C,4*C")
        fc2 = FakeTensor("4*C,C")
        x3 = x2.matmul(fc1).matmul(fc2)
        x3.show_compute_graph()
    else:
        router = FakeTensor("C,E")
        #score = x2.matmul(router).top_k("topK") # B,T,topK
        x3 = x2.repeat("topK", dim=2) # B,T,topK,C
        x4 = x3.view("B,T*topK,C").view("B,E,T*topK/E,C") #TODO:shuffle
        experts_fc1 = FakeTensor("E,C,Ce")
        experts_fc2 = FakeTensor("E,Ce,C")
        x5 = x4.matmul(experts_fc1).matmul(experts_fc2) # B,E,T*topK/E,C
        x6 = x5.view("B,T*topK,C").view("B,T,topK,C").sum(dim=-2)
        x6.show_compute_graph()
        

def multi_device_example():
    ulysses_style = False
    
    register_variable("B", 1)
    register_variable("T", 4096)
    register_variable("H", 96)
    register_variable("D", 128)
    register_variable("C", "D*H")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "C")
    register_variable("P", 8)

    #print(f"g_var_registry: {g_var_registry}\ng_merge_registry: {g_merge_registry}");

    x = FakeTensor("B,T/P,C", DeviceSpan("P"))
    all_x = FakeTensor("B,T/P,C", DeviceSpan("P"))
    Wq = FakeTensor("C,C", DeviceSpan("P")) # TODO: how to set DeviceSpan?
    Wk = FakeTensor("C,C", DeviceSpan("P"))
    Wv = FakeTensor("C,C", DeviceSpan("P"))
    q = x.matmul(Wq).view("B,T/P,H,D").transpose(1,2).all2all(gather_axis=2, scatter_axis=1)
    k = all_x.matmul(Wk).view("B,T/P,H,D").transpose(1,2).all2all(gather_axis=2, scatter_axis=1)
    v = all_x.matmul(Wv).view("B,T/P,H,D").transpose(1,2).all2all(gather_axis=2, scatter_axis=1)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    #print(f"q={q} k={k} v={v} attn={attn}")
    merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,T,C/P")
    if ulysses_style:
        merged_v = merged_v.all2all(gather_axis=2, scatter_axis=1) # B,T/P,C
        Wo = FakeTensor("C,C", DeviceSpan("P"))
        x2 = merged_v.matmul(Wo).all_gather(axis=1)
        fc1 = FakeTensor("C,4*C/P", DeviceSpan("P"))
        fc2 = FakeTensor("4*C/P,C", DeviceSpan("P"))
        x3 = x2.matmul(fc1).matmul(fc2).reduce_scatter(scatter_axis=1)
        x3.show_compute_graph()
    else:
        Wo = FakeTensor("C/P,C", DeviceSpan("P"))
        x2 = merged_v.matmul(Wo).all_reduce()
        fc1 = FakeTensor("C,4*C/P", DeviceSpan("P"))
        fc2 = FakeTensor("4*C/P,C", DeviceSpan("P"))
        x3 = x2.matmul(fc1).matmul(fc2).reduce_scatter(scatter_axis=1)
        x3.show_compute_graph()

def example2():
    register_variable("B", 1)
    register_variable("T", 1024)
    register_variable("H", 128)
    register_variable("D", 128)
    register_variable("C", 5120)
    register_variable("Dk", "0.1*C")
    register_variable("Dq", "0.3*C")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "3.2*C")
    register_variable("Tq", "T")

    #print(f"g_var_registry: {g_var_registry}\ng_merge_registry: {g_merge_registry}");
    
    cur_x = FakeTensor("B,Tq,C")
    all_x = FakeTensor("B,T,C")
    W_dq = FakeTensor("C,Dq")
    W_kv = FakeTensor("C,Dk")
    C_q = cur_x.matmul(W_dq)
    C_kv = all_x.matmul(W_kv)
    W_uk = FakeTensor("Dk,D*H")
    W_uv = FakeTensor("Dk,D*H")
    k = C_kv.matmul(W_uk).view("B,T,H,D").transpose(1,2)
    v = C_kv.matmul(W_uv).view("B,T,H,D").transpose(1,2)
    W_uq = FakeTensor("Dq,D*H")
    q = C_q.matmul(W_uq).view("B,Tq,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,Tq,D*H")
    Wo = FakeTensor("D*H,C")
    x2 = merged_v.matmul(Wo)
    
    x2.show_compute_graph()


if __name__ == '__main__':
    #examples = [test_graph, single_device_example, example2]
    examples = [single_device_example]

    for example in examples:
        example()
        Stats.print_stats()
        Stats.clear_stats()
        clear_variables()
