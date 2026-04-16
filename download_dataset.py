from modelscope import dataset_snapshot_download

clip_ids = [
    "00c117c-e1bb-4e8d-b49c-ff7482dc2aa5",
    "00c2b5a-043f-41ff-a154-faf130f8ba19",
    "00da9de-0ee5-465a-9a2d-e7e91d3016bb",
    "00f7979-5ad0-463c-8a22-a80fccd2ac7f",
    "014e0a0-4f11-4fd1-9db4-37cb9f4a20b9",
    "0158dd0-7fd3-4305-ba56-342717481c0f",
    "0177f2f-8a50-4fa4-ba38-b6b3fc96ee37",
    "019068b-5618-4a6c-93d4-89dc415c8a3a",
]

patterns = [f"clips/{cid}/*" for cid in clip_ids]

dataset_snapshot_download(
    dataset_id="nv-community/PhysicalAI-Autonomous-Vehicles-NCore",
    local_dir="./ncore_10clips",
    allow_file_pattern=patterns,
)