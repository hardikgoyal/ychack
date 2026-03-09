"""Pre-compute 3D structures for preset drugs and cache PDB files to disk."""

import hashlib
import os
import sys
import time

# Load env before imports that might need it
from dotenv import load_dotenv
load_dotenv()

from sequence_engine import parse_multi_fasta
from tamarind_integration import submit_fold_job, check_fold_status, fetch_fold_result

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".pdb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

API_KEY = os.environ.get("TAMARIND_API_KEY", "")

PRESETS = {
    "Odronextamab (CD20xCD3 bispecific)": """>Heavy Chain 1 (anti-CD20)
EVQLVESGGGLVQPGRSLRLSCVASGFTFNDYAMHWVRQAPGKGLEWVSVISWNSDSIGY
ADSVKGRFTISRDNAKNSLYLQMHSLRAEDTALYYCAKDNHYGSGSYYYYQYGMDVWGQG
TTVTVSSASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTF
PAVLQSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPPCPAPP
VAGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREE
QFNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPS
QEEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSRLTVDK
SRWQEGNVFSCSVMHEALHNHYTQKSLSLSLG
>Heavy Chain 2 (anti-CD3)
EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYTMHWVRQAPGKGLEWVSGISWNSGSIGY
ADSVKGRFTISRDNAKKSLYLQMNSLRAEDTALYYCAKDNSGYGHYYYGMDVWGQGTTVT
VASASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVL
QSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPPCPAPPVAGP
SVFLFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREEQFNS
TYRVVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPSQEEM
TKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSRLTVDKSRWQ
EGNVFSCSVMHEALHNRFTQKSLSLSLG
>Light Chain
EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASTRATGIPA
RFSGSGSGTEFTLTISSLQSEDFAVYYCQHYINWPLTFGGGTKVEIKRTVAAPSVFIFPP
SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT
LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC""",

    "Bococizumab (anti-PCSK9)": """>Heavy Chain
QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS""",

    "Moxetumomab pasudotox (scFv immunotoxin)": """>VL
DIQMTQTTSSLSASLGDRVTISCRASQDISKYLNWYQQKPDGTVKLLIYHTSRLHSGVPS
RFSGSGSGTDYSLTISNLEQEDIATYFCQQGNTLPYTFGGGTKLEIT
>Linker (Whitlow)
GSTSGSGKPGSGEGSTKG
>VH
EVKLQESGPGLVAPSQSLSVTCTVSGVSLPDYGVSWIRQPPRKGLEWLGVIWGSETTYYN
SALKSRLTIIKDNSKSQVFLKMNSLQTDDTAIYYCAKHYYYGGSYAMDYWGQGTSVTVSS""",
}


def cache_path_for(chain_name, chain_seq):
    seq_hash = hashlib.sha256(chain_seq.encode()).hexdigest()[:12]
    safe_name = chain_name.replace(" ", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_{seq_hash}.pdb")


def fold_chain(chain_name, chain_seq):
    """Submit fold job, poll until complete, return PDB string."""
    print(f"  [{chain_name}] {len(chain_seq)} aa — submitting fold job...")
    job_name = submit_fold_job(chain_seq, API_KEY)
    if not job_name:
        print(f"  [{chain_name}] FAILED to submit fold job")
        return None

    print(f"  [{chain_name}] job={job_name}, polling...")
    start = time.time()
    while time.time() - start < 300:  # 5 min timeout
        status = check_fold_status(job_name, API_KEY)
        elapsed = int(time.time() - start)
        if status == "complete":
            print(f"  [{chain_name}] complete in {elapsed}s, downloading...")
            pdb = fetch_fold_result(job_name, API_KEY)
            if pdb:
                return pdb
            print(f"  [{chain_name}] download failed")
            return None
        elif status == "failed":
            print(f"  [{chain_name}] FAILED after {elapsed}s")
            return None
        else:
            print(f"  [{chain_name}] {status}... ({elapsed}s)")
            time.sleep(10)

    print(f"  [{chain_name}] TIMED OUT after 300s")
    return None


def run_preset(preset_name, fasta_text):
    chains = parse_multi_fasta(fasta_text)
    print(f"\n{'='*60}")
    print(f"  {preset_name} — {len(chains)} chains")
    print(f"{'='*60}")

    for chain_name, chain_seq in chains.items():
        path = cache_path_for(chain_name, chain_seq)
        if os.path.exists(path):
            print(f"  [{chain_name}] already cached, skipping")
            continue

        pdb = fold_chain(chain_name, chain_seq)
        if pdb:
            with open(path, "w") as f:
                f.write(pdb)
            print(f"  [{chain_name}] saved to {os.path.basename(path)}")
        else:
            print(f"  [{chain_name}] no PDB obtained")

    print(f"  Done: {preset_name}\n")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: TAMARIND_API_KEY not set in .env or environment")
        sys.exit(1)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fasta in PRESETS.items():
        if target == "all" or target.lower() in name.lower():
            run_preset(name, fasta)
