"""Ручна промоція кандидатів еталонів: .venv/bin/python align_promote_cli.py [--apply РЕЖИМ]
Без аргументів — лише показує кандидатів і чи є консенсус (нічого не міняє). --apply РЕЖИМ — промоція (стара версія → data/refs_ver/<мітка>/prev).
Запускати, коли тихо (немає руху й подій). Службу перезапускати не треба: еталони читаються з диска на кожну подію."""
import sys
import time

import align_guard as ag

now = time.time()
poses = ag.load_ref_poses()
for mode in ("ir", "irday"):
    cands = ag.list_candidates(mode, now)
    group = ag.consensus(cands)
    print(f"{mode}: кандидатів {len(cands)}, консенсус: {'є (' + str(len(group)) + ')' if group else 'нема'}")
    if group:
        print("   збіг зі старим еталоном:", [ag.matches_old(c, mode, poses) for c in group])
if len(sys.argv) == 3 and sys.argv[1] == "--apply":
    mode = sys.argv[2]
    group = ag.consensus(ag.list_candidates(mode, now))
    if not group:
        sys.exit("консенсусу нема — промоції не буде")
    print(ag.promote(group, mode, now))
