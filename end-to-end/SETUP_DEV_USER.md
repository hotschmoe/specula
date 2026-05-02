# Recreating the `dev` user on a fresh VM

If `/workspace` mounts onto a clean RunPod with only `root` available,
this is the recipe to recreate the `dev` user that the e2e pipeline
expects (per RESUME.md). Run as root.

```bash
# 1. Create the user with home, bash, sudo group.
useradd -m -s /bin/bash -G sudo dev

# 2. Passwordless sudo.
echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-dev
chmod 0440 /etc/sudoers.d/90-dev

# 3. Mirror SSH access from root (RunPod populates root's
#    authorized_keys at boot from $PUBLIC_KEY env). Copying lets
#    `ssh dev@<host>` work too if needed.
mkdir -p /home/dev/.ssh
cp /root/.ssh/authorized_keys /home/dev/.ssh/authorized_keys 2>/dev/null || true
chmod 700 /home/dev/.ssh && chmod 600 /home/dev/.ssh/authorized_keys
chown -R dev:dev /home/dev/.ssh

# 4. Mirror gh CLI auth (the credential helper for git push).
mkdir -p /home/dev/.config/gh
cp /root/.config/gh/hosts.yml /home/dev/.config/gh/hosts.yml
chmod 600 /home/dev/.config/gh/hosts.yml
chown -R dev:dev /home/dev/.config

# 5. git identity for dev.
sudo -u dev bash -c 'git config --global user.name "hotschmoe"'
sudo -u dev bash -c 'git config --global user.email "stronggarner66@gmail.com"'
sudo -u dev bash -c '
git config --global credential.https://github.com.helper "" && \
git config --global --add credential.https://github.com.helper "!/usr/bin/gh auth git-credential" && \
git config --global credential.https://gist.github.com.helper "" && \
git config --global --add credential.https://gist.github.com.helper "!/usr/bin/gh auth git-credential"
'

# 6. Append env vars to dev's .bashrc so interactive shells get
#    aimet venv + qairt SDK + LD_LIBRARY_PATH for ORT-CUDA's
#    bundled NV libs.
cat >> /home/dev/.bashrc <<'EOF'

# ---- specula e2e env (added by setup-dev) ----
export AIMET_VENV=/workspace/venvs/aimet-2.26-cu121-py310
export QAIRT_SDK_ROOT=/workspace/sdks/qairt-2.45.40.260406
NVLIBS=$(find $AIMET_VENV/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${NVLIBS}${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}
export PATH=${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang:${AIMET_VENV}/bin:${PATH}
export PYTHONPATH=${QAIRT_SDK_ROOT}/lib/python:${PYTHONPATH:-}
alias activate-aimet='source $AIMET_VENV/bin/activate'
alias specula='cd /workspace/specula'
EOF
chown dev:dev /home/dev/.bashrc

# 7. Verify.
sudo -u dev bash -i -c '
  echo "PATH includes qairt: $(echo $PATH | grep -o qairt | head -1)"
  echo "AIMET_VENV: $AIMET_VENV"
  which python qairt-converter qnn-context-binary-generator
  python -c "import aimet_onnx, onnxruntime; print(aimet_onnx.__version__, onnxruntime.__version__)"
  cd /workspace/specula && git ls-remote origin HEAD | head -1
' 2>&1 | grep -v "ioctl\|job control\|administrator\|man sudo"
```

Expected verification output:

```
PATH includes qairt: qairt
AIMET_VENV: /workspace/venvs/aimet-2.26-cu121-py310
/workspace/venvs/aimet-2.26-cu121-py310/bin/python
/workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qairt-converter
/workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qnn-context-binary-generator
2.26.0+cu121 1.23.2
<sha>	HEAD
```

## Switching to dev

```bash
su - dev      # interactive bash as dev, env loaded
# or
sudo -u dev -i
```

## Notes

- `/workspace` is `drwxrwxrwx` so dev can read+write everything
  there without ownership changes.
- The aimet venv was built as root but the venv binaries don't
  care who runs them; they only need read/exec which is granted.
- Don't `chown -R dev:dev /workspace/specula` — the .git repo's
  loose objects might already be owned by root from prior commits.
  Mixed ownership is fine because permissions on /workspace are
  permissive.
