"""Pre-compute MHC-I cytotoxic analysis for preset drugs and cache to disk."""

import json
import os
import sys
from dataclasses import asdict
from sequence_engine import parse_multi_fasta
from safebind_mhc1_cytotoxic import run_cytotoxic_assessment

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".mhc1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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


def serialize_report(report):
    """Convert CytotoxicReport to JSON-serializable dict."""
    d = asdict(report)
    return d


def cache_key(chain_name, chain_seq):
    """Stable cache key from chain name + sequence hash."""
    import hashlib
    seq_hash = hashlib.sha256(chain_seq.encode()).hexdigest()[:12]
    safe_name = chain_name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}_{seq_hash}"


def run_preset(preset_name, fasta_text):
    chains = parse_multi_fasta(fasta_text)
    print(f"\n{'='*60}")
    print(f"  {preset_name} — {len(chains)} chains")
    print(f"{'='*60}")

    for chain_name, chain_seq in chains.items():
        key = cache_key(chain_name, chain_seq)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")

        if os.path.exists(cache_path):
            print(f"  [{chain_name}] already cached, skipping")
            continue

        print(f"  [{chain_name}] {len(chain_seq)} aa — running IEDB MHC-I...")
        report = run_cytotoxic_assessment(
            chain_seq, name=chain_name, use_mhcflurry=False, use_iedb=True,
        )
        data = serialize_report(report)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"  [{chain_name}] done — {report.total_epitopes_predicted} epitopes, "
              f"risk={report.overall_cytotoxic_risk:.2f} ({report.risk_category})")

    print(f"  Done: {preset_name}\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fasta in PRESETS.items():
        if target == "all" or target in name.lower():
            run_preset(name, fasta)
