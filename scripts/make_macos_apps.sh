#!/bin/bash
# Build macOS application bundles for the T1Prep viewers.
#
# The bundles are thin: each one launches the installed console entry point
# (CAT_SurfView / CAT_VolView) with the interpreter of the environment they
# were built from.  Nothing is copied or frozen, so the apps follow every
# update of the installation they point at — and they stop working if that
# environment is removed.
#
# Usage:
#   scripts/make_macos_apps.sh [-o <output-dir>] [-p <bin-dir>]
#
#   -o   where to write the .app bundles
#        (default: /Applications when writable, else ~/Applications)
#   -p   directory holding CAT_SurfView / CAT_VolView
#        (default: taken from PATH, else the project venv env/bin)
#   -d   also make the apps the default for the file types they declare
#        (needs duti: brew install duti)

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script builds macOS app bundles and only runs on macOS." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"

out_dir=""
bin_dir=""
set_default=0
while getopts ":o:p:dh" opt; do
    case "$opt" in
        o) out_dir="$OPTARG" ;;
        p) bin_dir="$OPTARG" ;;
        d) set_default=1 ;;
        h) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------- entry points
if [[ -z "$bin_dir" ]]; then
    if command -v CAT_SurfView >/dev/null 2>&1; then
        bin_dir="$(dirname "$(command -v CAT_SurfView)")"
    elif [[ -x "$project_dir/env/bin/CAT_SurfView" ]]; then
        bin_dir="$project_dir/env/bin"
    fi
fi
for entry in CAT_SurfView CAT_VolView; do
    if [[ -z "$bin_dir" || ! -x "$bin_dir/$entry" ]]; then
        echo "❌ Could not find $entry${bin_dir:+ in $bin_dir}." >&2
        echo "   Install T1Prep (pip install T1Prep) and activate the environment," >&2
        echo "   or point at its bin directory with -p <dir>." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- output
if [[ -z "$out_dir" ]]; then
    if [[ -w /Applications ]]; then out_dir="/Applications"; else out_dir="$HOME/Applications"; fi
fi
mkdir -p "$out_dir"

# ---------------------------------------------------------------- icon (optional)
# Rendered from the project logo when the macOS tools are available; the apps
# fall back to the generic icon otherwise.
icon_source="$project_dir/T1Prep_logo.svg"
icns=""
if [[ -f "$icon_source" ]] && command -v qlmanage >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    work="$(mktemp -d)"
    if qlmanage -t -s 1024 -o "$work" "$icon_source" >/dev/null 2>&1; then
        png="$(find "$work" -name '*.png' | head -1)"
        if [[ -n "$png" ]]; then
            iconset="$work/T1Prep.iconset"
            mkdir -p "$iconset"
            for size in 16 32 64 128 256 512; do
                sips -z "$size" "$size" "$png" --out "$iconset/icon_${size}x${size}.png" >/dev/null 2>&1 || true
                sips -z $((size * 2)) $((size * 2)) "$png" \
                     --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null 2>&1 || true
            done
            if iconutil -c icns "$iconset" -o "$work/T1Prep.icns" >/dev/null 2>&1; then
                icns="$work/T1Prep.icns"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------- bundles
# Finder needs more than a file extension to route documents: it matches on
# Uniform Type Identifiers.  ".nii" already has one on macOS
# (gov.nih.nifti-1), which we import; ".gii"/".annot" have none, so the apps
# export their own.  ".nii.gz" is seen as plain gzip — only the last extension
# counts — so the volume viewer registers for gzip as an *alternate* handler:
# it then shows up under "Open With" without claiming every .gz file.
make_app() {
    local name="$1" description="$2" doc_types="$3" declarations="$4"
    local app="$out_dir/$name.app"

    rm -rf "$app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

    # Finder discards stdout/stderr, so a crash would be silent; keep the last
    # run in a log the user can be pointed at.
    cat > "$app/Contents/MacOS/$name" <<LAUNCHER
#!/bin/bash
# Launcher for $name; regenerate with scripts/make_macos_apps.sh
export T1PREP_APP=1
log_dir="\$HOME/Library/Logs/T1Prep"
mkdir -p "\$log_dir"
exec "$bin_dir/$name" "\$@" > "\$log_dir/$name.log" 2>&1
LAUNCHER
    chmod +x "$app/Contents/MacOS/$name"

    local icon_entry=""
    if [[ -n "$icns" ]]; then
        cp "$icns" "$app/Contents/Resources/$name.icns"
        icon_entry="    <key>CFBundleIconFile</key>
    <string>$name</string>"
    fi

    cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$name</string>
    <key>CFBundleDisplayName</key>
    <string>$name</string>
    <key>CFBundleGetInfoString</key>
    <string>$description</string>
    <key>CFBundleExecutable</key>
    <string>$name</string>
    <key>CFBundleIdentifier</key>
    <string>de.uni-jena.t1prep.$(echo "$name" | tr '[:upper:]' '[:lower:]')</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$version</string>
    <key>CFBundleVersion</key>
    <string>$version</string>
$icon_entry
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.medical</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
$doc_types    </array>
$declarations</dict>
</plist>
PLIST

    plutil -lint "$app/Contents/Info.plist" >/dev/null
    echo "✅ $app"
}

# A CFBundleDocumentTypes entry
doc_type() {   # name, rank, UTIs (space separated), extensions (space separated)
    local content_types="" extensions=""
    for uti in $3; do content_types+="                <string>$uti</string>
"; done
    for ext in $4; do extensions+="                <string>$ext</string>
"; done
    cat <<ENTRY
        <dict>
            <key>CFBundleTypeName</key>
            <string>$1</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>$2</string>
            <key>LSItemContentTypes</key>
            <array>
$content_types            </array>
            <key>CFBundleTypeExtensions</key>
            <array>
$extensions            </array>
        </dict>
ENTRY
}

# A UTI declaration for a type nothing else on the system declares
exported_type() {   # identifier, description, extension, conforms-to
    cat <<ENTRY
        <dict>
            <key>UTTypeIdentifier</key>
            <string>$1</string>
            <key>UTTypeDescription</key>
            <string>$2</string>
            <key>UTTypeConformsTo</key>
            <array><string>$4</string></array>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array><string>$3</string></array>
            </dict>
        </dict>
ENTRY
}

version="$(awk -F\" '/^__version__/ {print $2}' "$project_dir/src/t1prep/__init__.py" 2>/dev/null || echo "0.0")"

surf_docs="$(doc_type "GIFTI surface" Owner "de.uni-jena.t1prep.gifti" "gii")
$(doc_type "FreeSurfer annotation" Owner "de.uni-jena.t1prep.annot" "annot")
$(doc_type "Surface overlay" Alternate "public.plain-text" "txt")"
surf_utis="    <key>UTExportedTypeDeclarations</key>
    <array>
$(exported_type "de.uni-jena.t1prep.gifti" "GIFTI surface" "gii" "public.xml")
$(exported_type "de.uni-jena.t1prep.annot" "FreeSurfer annotation" "annot" "public.data")
    </array>
"

vol_docs="$(doc_type "NIfTI volume" Default "gov.nih.nifti-1" "nii")
$(doc_type "Compressed volume" Alternate "org.gnu.gnu-zip-archive" "gz")
$(doc_type "Volume" Alternate "public.data" "mnc nrrd mha mhd")"
vol_utis="    <key>UTImportedTypeDeclarations</key>
    <array>
$(exported_type "gov.nih.nifti-1" "NIfTI volume" "nii" "public.data")
    </array>
"

make_app "CAT_SurfView" "T1Prep surface viewer" "$surf_docs" "$surf_utis"
make_app "CAT_VolView"  "T1Prep volume viewer"  "$vol_docs" "$vol_utis"

# Finder caches bundle metadata; nudge it so the new apps are picked up
if [[ -x /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister ]]; then
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
        -f "$out_dir/CAT_SurfView.app" "$out_dir/CAT_VolView.app" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------- defaults
# Which app opens a type by default is a user setting, not something a bundle
# can decide; duti writes it, otherwise Finder does.
surf_id="de.uni-jena.t1prep.cat_surfview"
vol_id="de.uni-jena.t1prep.cat_volview"
if [[ "$set_default" -eq 1 ]]; then
    if command -v duti >/dev/null 2>&1; then
        duti -s "$vol_id"  gov.nih.nifti-1              all || true
        duti -s "$vol_id"  nii                          all || true
        duti -s "$surf_id" de.uni-jena.t1prep.gifti     all || true
        duti -s "$surf_id" gii                          all || true
        duti -s "$surf_id" de.uni-jena.t1prep.annot     all || true
        echo "✅ Default application set for .nii, .gii and .annot"
        echo "   (.nii.gz is a gzip archive to macOS — see below)"
    else
        echo "⚠ duti is not installed, so the defaults were not changed." >&2
        echo "  Install it with 'brew install duti' and re-run with -d," >&2
        echo "  or set it once in Finder (see below)." >&2
    fi
fi

echo ""
echo "The apps launch $bin_dir/CAT_SurfView and CAT_VolView."
echo "Double-click one to pick a file, or drop files onto its icon."
echo ""
echo "To always open a file type with them, select a file in Finder,"
echo "press ⌘I, choose the app under \"Open with\" and click \"Change All…\"."
echo "Do this once for .nii and once for .nii.gz: macOS only looks at the last"
echo "extension, so .nii.gz counts as a gzip archive."
