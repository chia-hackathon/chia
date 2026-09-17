#!/usr/bin/env python3
"""
Add spike-custom-extension support to chipyard's cospike.

generators/testchipip/src/main/resources/testchipip/csrc/cospike_impl.cc builds its
own cfg_t and sim_t.  cfg_t has no extension list -- spike takes custom extensions
from the ISA string ("..._x<name>", see disasm/isa_parser.cc:412), and processor_t's
constructor then does

    for (auto e : isa.get_extensions())
      register_extension(find_extension(e.c_str())());

find_extension() (riscv/extensions.cc) already dlopen()s lib<name>.so itself when the
name is not registered yet, so no dlopen is strictly required on our side.  Doing the
registration through the ISA string (rather than calling processor_t::register_extension
after construction) is what gets the ordering right: the extension's CSRs are added and
its reset() is called from processor_t::reset(), which runs inside the constructor.

This patch adds two opt-in plusargs (plus equivalent env vars).  Default is empty, so
behaviour for configs without a custom accelerator is bit-identical to before.

  +cospike-extension=<name>[,<name>...]   append "_x<name>" to the cosim ISA string
                                          (env: COSPIKE_EXTENSIONS)
  +cospike-extlib=<path.so>               dlopen an extension library explicitly, for
                                          libraries not on the loader search path
                                          (env: COSPIKE_EXTLIBS, comma separated)

e.g. Gemmini:  +cospike-extension=gemmini   ->  isa "..._zicntr_xgemmini",
               find_extension("gemmini") dlopens libgemmini.so from $RISCV/lib.

Deterministic exact-string replacement (no line numbers involved).
"""
import sys

path = sys.argv[1]
src = open(path).read()


def sub(anchor, new, expect=1):
    global src
    n = src.count(anchor)
    if n != expect:
        sys.exit("FATAL: expected %d match(es) for anchor in %s, found %d\n---\n%s"
                 % (expect, path, n, anchor))
    src = src.replace(anchor, new)


# 1) includes
sub("""#include <riscv/mmu.h>""",
    """#include <riscv/mmu.h>
#include <riscv/extension.h>
#include <dlfcn.h>
#include <cstdlib>""")

# 2) globals
sub("""reg_t cospike_timeout = 0;""",
    """reg_t cospike_timeout = 0;
// Custom spike extensions (RoCC accelerator models such as Gemmini's).  Empty by
// default: a target without a custom accelerator behaves exactly as before.
std::vector<std::string> cospike_extensions;
std::vector<std::string> cospike_extlibs;

// Split "a,b,c" (also accepts ':' so LD_LIBRARY_PATH-style lists work).
static void cospike_split_list(const std::string &s, std::vector<std::string> &out) {
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find_first_of(",:", start);
    if (end == std::string::npos) end = s.size();
    if (end > start) out.push_back(s.substr(start, end - start));
    start = end + 1;
  }
}""")

# 3) plusargs
sub("""      } else if (arg.find("+cospike-enable=") == 0) {
\tcospike_enable = strtoull(arg.substr(16).c_str(), 0, 10) != 0;
      } else if (!in_permissive) {""",
    """      } else if (arg.find("+cospike-enable=") == 0) {
\tcospike_enable = strtoull(arg.substr(16).c_str(), 0, 10) != 0;
      } else if (arg.find("+cospike-extension=") == 0) {
        cospike_split_list(arg.substr(19), cospike_extensions);
      } else if (arg.find("+cospike-extlib=") == 0) {
        cospike_split_list(arg.substr(16), cospike_extlibs);
      } else if (!in_permissive) {""")

# 4) act on them, at the end of cospike_set_sysinfo (info->isa is finalized here and
#    read by cfg->isa later; cfg_t stores the char* so it must not change afterwards)
sub("""      } else if (!in_permissive) {
        info->htif_args.push_back(arg);
      }
    }
  }
}""",
    """      } else if (!in_permissive) {
        info->htif_args.push_back(arg);
      }
    }

    // Env-var fallback, for flows where adding plusargs is inconvenient.
    if (const char* e = getenv("COSPIKE_EXTENSIONS")) cospike_split_list(e, cospike_extensions);
    if (const char* e = getenv("COSPIKE_EXTLIBS"))    cospike_split_list(e, cospike_extlibs);

    // Explicitly requested libraries first.  RTLD_GLOBAL so the extension's
    // REGISTER_EXTENSION static initializer can bind against libriscv.
    for (const auto& lib : cospike_extlibs) {
      void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!h) {
        COSPIKE_PRINTF("Unable to load spike extlib '%s': %s\\n", lib.c_str(), dlerror());
        abort();
      }
      COSPIKE_PRINTF("Loaded spike extlib %s\\n", lib.c_str());
    }

    // Custom extensions are requested through the ISA string: processor_t's
    // constructor calls find_extension() for every "x" extension it parses, which
    // registers the instructions/disasm/CSRs *before* the initial reset().
    // find_extension() dlopens lib<name>.so on its own if it is not registered yet.
    for (const auto& ext : cospike_extensions) {
      std::string tok = "_x" + ext;
      if (info->isa.find(tok) == std::string::npos) info->isa += tok;
      COSPIKE_PRINTF("Enabling spike custom extension '%s'\\n", ext.c_str());
    }
  }
}""")

open(path, "w").write(src)
print("PATCHED %s" % path)
