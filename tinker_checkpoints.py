import tinker
rc = tinker.ServiceClient().create_rest_client()

ckpts, offset = [], 0
while True:
    resp = rc.list_user_checkpoints(limit=100, offset=offset).result()
    ckpts += resp.checkpoints
    if len(resp.checkpoints) < 100:
        break
    offset += 100

for c in sorted(ckpts, key=lambda c: c.time):
    print(f"{c.time}  {c.checkpoint_type:9}  {c.size_bytes/1e6:7.0f} MB  expires={c.expires_at}  {c.tinker_path}")
print(f"\n{len(ckpts)} checkpoints total")
