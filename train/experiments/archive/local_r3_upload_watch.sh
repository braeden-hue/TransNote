#!/bin/bash
cd /c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/round3train
POOL=data/local_pools/r3_density_register_clef
while :; do
  N=$(find "$POOL" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$N" -ge 5000 ]; then break; fi
  sleep 20
done
echo "[r3upload] 로컬 Round3 5000장 완료 감지 -- 압축 시작"
cd data/local_pools
tar -czf r3_density_register_clef.tar.gz r3_density_register_clef
echo "[r3upload] 압축 완료 -- pod로 업로드 시작"
scp -i ~/.ssh/runpod_auto -P 11094 -o StrictHostKeyChecking=no \
  r3_density_register_clef.tar.gz \
  root@213.173.108.16:/workspace/r3_density_register_clef.tar.gz
echo "[r3upload] 업로드 완료 -- pod에서 압축 해제 시작"
ssh -i ~/.ssh/runpod_auto -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@213.173.108.16 -p 11094 \
  "mkdir -p /workspace/data && cd /workspace/data && tar -xzf /workspace/r3_density_register_clef.tar.gz && rm /workspace/r3_density_register_clef.tar.gz && find /workspace/data/r3_density_register_clef -maxdepth 1 -name '*.png' | wc -l"
echo "[r3upload] pod 반영 완료"
