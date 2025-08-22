import os
import pandas as pd
from tqdm import tqdm

main_dir = "PATH TO YOUR DATASET"
parquet_path = "PATH TO YOUR NODE EMBEDDINGS FILE"

df = pd.read_parquet(parquet_path)
embedding_dict = dict(zip(df['uuid'].astype(str), df['embedding']))

valid_assemblies = []
assembly_parts_count = {}

for assembly in tqdm(os.listdir(main_dir), desc="Checking assemblies"):
    assembly_path = os.path.join(main_dir, assembly)
    if not os.path.isdir(assembly_path):
        continue
    part_uuids = [
        os.path.splitext(f)[0]
        for f in os.listdir(assembly_path)
        if f.endswith('.smt') and not f.startswith("assembly.")
    ]
    # Check if all parts have a non-None embedding
    if part_uuids and all(
        (uuid in embedding_dict and embedding_dict[uuid] is not None)
        for uuid in part_uuids
    ):
        valid_assemblies.append(assembly)
        assembly_parts_count[assembly] = len(part_uuids)

print(f"Total valid assemblies: {len(valid_assemblies)}")
valid_assemblies_df = pd.DataFrame({'assembly_id': valid_assemblies})

valid_assemblies_df.to_parquet('PATH TO YOUR DATASET/valid_assemblies.parquet', index=False)
print("Saved valid assembly names to PATH TO YOUR DATASET/valid_assemblies.parquet")

print("Assemblies and number of valid parts:")
for assembly, count in assembly_parts_count.items():
    print(f"{assembly}: {count} parts")