#!/usr/bin/env bash
set -Eeuo pipefail

ADB="${ADB:-adb}"

if ! "$ADB" get-state >/dev/null 2>&1; then
  echo "No authorized Android device. Connect USB, enable Developer options and USB debugging, then accept the RSA prompt." >&2
  exit 2
fi

getprop_value() {
  "$ADB" shell getprop "$1" | tr -d '\r'
}

echo "manufacturer=$(getprop_value ro.product.manufacturer)"
echo "model=$(getprop_value ro.product.model)"
echo "device=$(getprop_value ro.product.device)"
echo "product=$(getprop_value ro.product.name)"
echo "soc_model=$(getprop_value ro.soc.model)"
echo "soc_manufacturer=$(getprop_value ro.soc.manufacturer)"
echo "hardware=$(getprop_value ro.hardware)"
echo "board_platform=$(getprop_value ro.board.platform)"
echo "android_release=$(getprop_value ro.build.version.release)"
echo "sdk=$(getprop_value ro.build.version.sdk)"
echo "abi=$(getprop_value ro.product.cpu.abi)"
echo "abilist=$(getprop_value ro.product.cpu.abilist)"
echo "build_fingerprint=$(getprop_value ro.build.fingerprint)"
ram_kib="$($ADB shell cat /proc/meminfo | tr -d '\r' | sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p')"
echo "ram_kib=$ram_kib"
echo "cpu_present=$($ADB shell cat /sys/devices/system/cpu/present 2>/dev/null | tr -d '\r')"
echo "cpu_possible=$($ADB shell cat /sys/devices/system/cpu/possible 2>/dev/null | tr -d '\r')"
echo "cpu_online=$($ADB shell cat /sys/devices/system/cpu/online 2>/dev/null | tr -d '\r')"
echo "cpu_count=$($ADB shell 'grep -c "^processor" /proc/cpuinfo' | tr -d '\r')"
echo "cpu_policy_configuration:"
"$ADB" shell '
for policy in /sys/devices/system/cpu/cpufreq/policy[0-9]*; do
  [ -d "$policy" ] || continue
  printf "policy=%s cpus=%s min_khz=%s max_khz=%s governor=%s\n" \
    "${policy##*/}" \
    "$(cat "$policy/related_cpus" 2>/dev/null)" \
    "$(cat "$policy/cpuinfo_min_freq" 2>/dev/null)" \
    "$(cat "$policy/cpuinfo_max_freq" 2>/dev/null)" \
    "$(cat "$policy/scaling_governor" 2>/dev/null)"
done' | tr -d '\r'
echo "cpu_capacity:"
"$ADB" shell '
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
  [ -d "$cpu" ] || continue
  printf "%s capacity=%s max_khz=%s\n" \
    "${cpu##*/}" \
    "$(cat "$cpu/cpu_capacity" 2>/dev/null)" \
    "$(cat "$cpu/cpufreq/cpuinfo_max_freq" 2>/dev/null)"
done' | tr -d '\r'
echo "cpuinfo:"
"$ADB" shell cat /proc/cpuinfo | tr -d '\r'
echo "thermal_zones=$($ADB shell 'ls /sys/class/thermal 2>/dev/null | grep -c thermal_zone || true' | tr -d '\r')"
echo "soc_dsp_properties:"
"$ADB" shell getprop | grep -Ei '\[(ro\.(soc|board|hardware)|ro\.vendor\..*(soc|dsp|adsp|cdsp|htp|hexagon|qnn)|vendor\..*(dsp|adsp|cdsp|htp|hexagon|qnn))' | tr -d '\r' || true
echo "qnn_libraries:"
"$ADB" shell 'find /vendor /system /system_ext /odm /apex -maxdepth 6 -type f \( -iname "*QnnHtp*" -o -iname "libQnn*.so" -o -iname "libhexagon*.so" \) 2>/dev/null | head -n 80' | tr -d '\r'
echo "htp_architecture_candidates:"
"$ADB" shell 'find /vendor /system /system_ext /odm /apex -maxdepth 7 -type f \( -iname "*QnnHtpV*Skel.so" -o -iname "*QnnHtpV*Stub.so" -o -iname "*hexagon*v*.so" \) 2>/dev/null | sed -n "1,120p"' | tr -d '\r'
echo "dsp_firmware_and_config:"
"$ADB" shell 'find /vendor/firmware /vendor/etc /odm/etc -maxdepth 5 \( -iname "*cdsp*" -o -iname "*adsp*" -o -iname "*hexagon*" -o -iname "*qnn*" -o -iname "*htp*" \) 2>/dev/null | head -n 120' | tr -d '\r'
