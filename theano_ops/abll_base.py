from theano.sandbox.cuda import GpuOp
from .abll_compile import is_available, loc, include_dir


class ABLLOp(GpuOp):
    def c_header_dirs(self):
        return [include_dir]

    def c_code_cache_version(self):
        raise NotImplementedError()

    def c_lib_dirs(self):
        return [loc]

    def c_libraries(self):
        return ['abll']

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        if not is_available():
            raise RuntimeError('Could not compile abll.')
        return super(ABLLOp, self).make_thunk(node, storage_map,
                                              compute_map, no_recycling)
