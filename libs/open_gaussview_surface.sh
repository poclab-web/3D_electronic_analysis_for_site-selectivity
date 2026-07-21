#!/bin/zsh
set -eu

if (( $# < 2 )); then
    print -u2 "Usage: $0 CUBE_PATH {electronic|electrostatic|orbital} [ISOVALUE]"
    exit 2
fi

cube_path="$1"
descriptor="$2"
override="${3:-}"

# Contribution cubes have the same units, so a fixed threshold permits direct
# visual comparison between substrates and descriptor blocks. The displayed
# full-y grid contains half of each folded |y| contribution on either face.
case "$descriptor" in
    electronic|electrostatic|orbital)
        isovalue="0.020"
        ;;
    *)
        print -u2 "Unknown descriptor block: $descriptor"
        exit 2
        ;;
esac
[[ -n "$override" ]] && isovalue="$override"

script_dir="${0:A:h}"
[[ -f "$cube_path" ]] || {
    print -u2 "Cube file does not exist: $cube_path"
    exit 1
}
gaussview_app="${GAUSSVIEW_APP:-GaussView}"
/usr/bin/open -a "$gaussview_app" "${cube_path:A}"
exec /usr/bin/osascript \
    "$script_dir/gaussview_quick_surface.applescript" \
    "${cube_path:A}" \
    "$isovalue"
