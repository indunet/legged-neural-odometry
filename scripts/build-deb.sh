#!/usr/bin/env bash
# Build dist/legged-neural-odometry_<version>_<arch>.deb for ROS 2 Humble.
# The package installs under /opt/legged-neural-odometry and enables
# legged-neural-odometry.service. Remove it with: dpkg -r legged-neural-odometry
#
# Environment:
#   ORT_DYLIB_PATH   path to libonnxruntime.so (1.31.x); auto-detected under
#                    /opt/onnxruntime/onnxruntime-linux-<arch>-1.31.0/lib otherwise
#   EXPECT_VERSION   if set (e.g. 1.2.3 from tag v1.2.3), fail unless all package versions match
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml | head -1)"
if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "build-deb: could not read a numeric version from pyproject.toml" >&2
  exit 1
fi
echo "build-deb: VERSION=${VERSION}" >&2

CARGO_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' ros2/legged_neural_odometry/Cargo.toml | head -1)"
PKG_XML_VERSION="$(sed -n 's|.*<version>\([^<]*\)</version>.*|\1|p' ros2/legged_neural_odometry/package.xml | head -1)"
for v in "${CARGO_VERSION}" "${PKG_XML_VERSION}" "${EXPECT_VERSION:-${VERSION}}"; do
  if [[ "${v}" != "${VERSION}" ]]; then
    echo "build-deb: version mismatch (pyproject=${VERSION}, Cargo.toml=${CARGO_VERSION}, package.xml=${PKG_XML_VERSION}, expected=${EXPECT_VERSION:-${VERSION}})" >&2
    exit 1
  fi
done

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "build-deb: missing /opt/ros/humble/setup.bash" >&2
  exit 1
fi
# ROS setup scripts read optional variables.
set +u
# Drop any previously sourced workspace so the bundled unitree_hg is the one that gets built in.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
set -u

if [[ -d "${HOME}/.cargo/bin" ]]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "build-deb: cargo is required" >&2
  exit 1
fi

BASE="${ROOT}/.deb-build"
rm -rf "${BASE}"
colcon --log-base "${BASE}/log" build \
  --base-paths "${ROOT}/ros2" \
  --build-base "${BASE}/build" \
  --install-base "${BASE}/install" \
  --merge-install \
  --packages-select unitree_hg legged_neural_odometry \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

BIN="${BASE}/install/lib/legged_neural_odometry/legged_neural_odometry"
MODEL="${BASE}/install/share/legged_neural_odometry/models/estimator.onnx"
if [[ ! -x "${BIN}" || ! -f "${MODEL}" ]]; then
  echo "build-deb: missing ${BIN} or ${MODEL}" >&2
  exit 1
fi

case "$(uname -m)" in
  aarch64) ORT_ARCH=aarch64 ;;
  x86_64) ORT_ARCH=x64 ;;
  *) echo "build-deb: unsupported machine $(uname -m)" >&2; exit 1 ;;
esac
ORT_DIR="/opt/onnxruntime/onnxruntime-linux-${ORT_ARCH}-1.31.0/lib"
ONNX_LIB=""
for candidate in \
  "${ORT_DYLIB_PATH:-}" \
  "${ORT_DIR}/libonnxruntime.so.1.31.0" \
  "${ORT_DIR}/libonnxruntime.so" \
  /usr/lib/libonnxruntime.so; do
  if [[ -n "${candidate}" && -f "${candidate}" ]]; then
    ONNX_LIB="${candidate}"
    break
  fi
done
if [[ -z "${ONNX_LIB}" ]]; then
  echo "build-deb: libonnxruntime 1.31 not found" >&2
  exit 1
fi

ARCH="$(dpkg --print-architecture)"
STAGE="$(mktemp -d)"
cleanup() { rm -rf "${STAGE}"; }
trap cleanup EXIT

PREFIX="opt/legged-neural-odometry"
mkdir -p "${STAGE}/DEBIAN" "${STAGE}/${PREFIX}" "${STAGE}/lib/systemd/system"
cp -a "${BASE}/install/." "${STAGE}/${PREFIX}/"
install -d "${STAGE}/${PREFIX}/lib"
SONAME="$(basename "${ONNX_LIB}")"
if [[ "${SONAME}" == libonnxruntime.so ]]; then
  SONAME="libonnxruntime.so.1.31.0"
fi
install -m 0644 "${ONNX_LIB}" "${STAGE}/${PREFIX}/lib/${SONAME}"
ln -sf "${SONAME}" "${STAGE}/${PREFIX}/lib/libonnxruntime.so.1"
ln -sf "${SONAME}" "${STAGE}/${PREFIX}/lib/libonnxruntime.so"
PROVIDERS="$(dirname "${ONNX_LIB}")/libonnxruntime_providers_shared.so"
if [[ -f "${PROVIDERS}" ]]; then
  install -m 0644 "${PROVIDERS}" "${STAGE}/${PREFIX}/lib/libonnxruntime_providers_shared.so"
fi

cat >>"${STAGE}/${PREFIX}/setup.bash" <<'EOF'
export LD_LIBRARY_PATH="/opt/legged-neural-odometry/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ORT_DYLIB_PATH="/opt/legged-neural-odometry/lib/libonnxruntime.so"
EOF

cat >"${STAGE}/lib/systemd/system/legged-neural-odometry.service" <<'EOF'
[Unit]
Description=Unitree G1 neural planar odometry
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/legged-neural-odometry
Environment=ORT_DYLIB_PATH=/opt/legged-neural-odometry/lib/libonnxruntime.so
Environment=LD_LIBRARY_PATH=/opt/legged-neural-odometry/lib
ExecStart=/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /opt/legged-neural-odometry/setup.bash && exec /opt/legged-neural-odometry/lib/legged_neural_odometry/legged_neural_odometry --ros-args -r __node:=legged_neural_odometry -p model_path:=/opt/legged-neural-odometry/share/legged_neural_odometry/models/estimator.onnx'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

BASE_DEPENDS="ros-humble-ros-workspace, ros-humble-nav-msgs, ros-humble-geometry-msgs, ros-humble-std-msgs, ros-humble-std-srvs, ros-humble-builtin-interfaces, ros-humble-rcl, ros-humble-rcl-action, ros-humble-rcutils, ros-humble-rmw, ros-humble-rmw-implementation, ros-humble-rosidl-runtime-c, ros-humble-rosidl-typesupport-c, ros-humble-rcl-interfaces, ros-humble-rosgraph-msgs, ros-humble-action-msgs, ros-humble-unique-identifier-msgs, ros-humble-rcl-logging-spdlog, ros-humble-libyaml-vendor, ros-humble-rosidl-default-runtime, ros-humble-rosidl-typesupport-fastrtps-c, ros-humble-rosidl-typesupport-fastrtps-cpp, ros-humble-rosidl-typesupport-introspection-c, ros-humble-rosidl-typesupport-introspection-cpp"
SHLIBS=""
if command -v dpkg-shlibdeps >/dev/null 2>&1; then
  SHLIB_DIR="$(mktemp -d)"
  mkdir -p "${SHLIB_DIR}/debian"
  printf 'Source: legged-neural-odometry\nPackage: legged-neural-odometry\n' >"${SHLIB_DIR}/debian/control"
  SHLIBS="$(cd "${SHLIB_DIR}" && dpkg-shlibdeps -O "${BIN}" 2>/dev/null | sed 's/^shlibs:Depends=//' || true)"
  rm -rf "${SHLIB_DIR}"
  SHLIBS="$(python3 - "${SHLIBS}" <<'PY'
import sys
parts = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
parts = [part for part in parts if "onnxruntime" not in part.lower()]
print(", ".join(parts))
PY
)"
fi
if [[ -n "${SHLIBS}" ]]; then
  PKG_DEPENDS="${BASE_DEPENDS}, ${SHLIBS}"
else
  PKG_DEPENDS="${BASE_DEPENDS}"
fi

cat >"${STAGE}/DEBIAN/control" <<EOF
Package: legged-neural-odometry
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Deng Ran <deng_ran@aliyun.com>
Depends: ${PKG_DEPENDS}
Description: Unitree G1 neural planar odometry
 Subscribes to /lowstate and publishes nav_msgs/Odometry on /g1/odom.
 Installed under /opt/legged-neural-odometry. The systemd unit
 legged-neural-odometry.service starts on install.
EOF

cat >"${STAGE}/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if [ "$1" != "configure" ]; then
  exit 0
fi
if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
  exit 0
fi
systemctl daemon-reload
if [ -n "${2:-}" ]; then
  systemctl enable legged-neural-odometry.service
  systemctl restart legged-neural-odometry.service
else
  systemctl enable --now legged-neural-odometry.service
fi
EOF

cat >"${STAGE}/DEBIAN/prerm" <<'EOF'
#!/bin/sh
set -e
if [ "$1" != "remove" ]; then
  exit 0
fi
if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
  exit 0
fi
systemctl disable --now legged-neural-odometry.service || true
EOF

cat >"${STAGE}/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
if [ "$1" != "remove" ] && [ "$1" != "purge" ]; then
  exit 0
fi
if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  systemctl daemon-reload || true
fi
EOF

chmod 0755 "${STAGE}/DEBIAN/postinst" "${STAGE}/DEBIAN/prerm" "${STAGE}/DEBIAN/postrm"

DEB_NAME="legged-neural-odometry_${VERSION}_${ARCH}.deb"
mkdir -p "${ROOT}/dist"
fakeroot dpkg-deb --root-owner-group -Zxz -z9 --build "${STAGE}" "${ROOT}/dist/${DEB_NAME}"
echo "Wrote ${ROOT}/dist/${DEB_NAME}" >&2
