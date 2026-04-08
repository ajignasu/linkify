#!/bin/bash
# Fusion 360 Assembly Dataset Download and Extraction Script
# This script automates the download, verification, and extraction of the dataset

set -e  # Exit on error

# Configuration
BASE_URL="https://fusion-360-gallery-assembly-interfaces.s3.us-west-2.amazonaws.com/public-archives"
PARTS="aa ab ac ad ae af ag ah ai aj"
CHECKSUM_FILE="contacts_assembly_json.tar.gz.sha256"
OUTPUT_DIR="contacts_assembly_json"
CLEANUP=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cleanup    Remove archive parts after successful extraction"
            echo "  --help       Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Download all dataset archive parts (~93.2 GB)"
            echo "  2. Download and verify SHA-256 checksums"
            echo "  3. Extract the dataset (~211 GB)"
            echo "  4. Optionally clean up archive parts"
            echo ""
            echo "Storage requirements:"
            echo "  - During download: ~93.2 GB"
            echo "  - During extraction: ~304 GB (archives + extracted data)"
            echo "  - After cleanup: ~211 GB (extracted data only)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check available disk space
print_info "Checking available disk space..."
AVAILABLE_SPACE=$(df -k . | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((320 * 1024 * 1024))  # 320 GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_warning "Low disk space detected"
    echo "Available: $(($AVAILABLE_SPACE / 1024 / 1024)) GB"
    echo "Recommended: 320 GB"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dataset already exists
if [ -d "$OUTPUT_DIR" ]; then
    print_warning "Dataset directory '$OUTPUT_DIR' already exists"
    read -p "Remove and re-extract? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing directory..."
        rm -rf "$OUTPUT_DIR"
    else
        print_info "Skipping extraction"
        exit 0
    fi
fi

# Determine download tool
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c"
    print_info "Using wget for downloads"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -C - -O"
    print_info "Using curl for downloads"
else
    print_error "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Step 1: Download checksum file
print_info "Step 1/4: Downloading checksum file..."
if [ ! -f "$CHECKSUM_FILE" ]; then
    if [[ $DOWNLOAD_CMD == wget* ]]; then
        wget -c "${BASE_URL}/${CHECKSUM_FILE}"
    else
        curl -C - -O "${BASE_URL}/${CHECKSUM_FILE}"
    fi
else
    print_info "Checksum file already exists, skipping download"
fi

# Step 2: Download all archive parts
print_info "Step 2/4: Downloading archive parts (this may take a while)..."
TOTAL_PARTS=10
CURRENT_PART=0

for part in $PARTS; do
    CURRENT_PART=$((CURRENT_PART + 1))
    FILENAME="contacts_assembly_json.tar.gz.part${part}"

    if [ -f "$FILENAME" ]; then
        print_info "[$CURRENT_PART/$TOTAL_PARTS] $FILENAME already exists, verifying..."
        # Quick size check
        if [ -s "$FILENAME" ]; then
            print_info "[$CURRENT_PART/$TOTAL_PARTS] $FILENAME looks valid, skipping download"
            continue
        fi
    fi

    print_info "[$CURRENT_PART/$TOTAL_PARTS] Downloading $FILENAME..."
    if [[ $DOWNLOAD_CMD == wget* ]]; then
        wget -c "${BASE_URL}/${FILENAME}"
    else
        curl -C - -O "${BASE_URL}/${FILENAME}"
    fi
done

print_info "All parts downloaded successfully"

# Step 3: Verify checksums
print_info "Step 3/4: Verifying checksums..."
if command -v shasum &> /dev/null; then
    if shasum -c "$CHECKSUM_FILE"; then
        print_info "All checksums verified successfully"
    else
        print_error "Checksum verification failed"
        print_error "One or more files may be corrupted. Please re-download the failing parts."
        exit 1
    fi
elif command -v sha256sum &> /dev/null; then
    if sha256sum -c "$CHECKSUM_FILE"; then
        print_info "All checksums verified successfully"
    else
        print_error "Checksum verification failed"
        print_error "One or more files may be corrupted. Please re-download the failing parts."
        exit 1
    fi
else
    print_warning "Neither shasum nor sha256sum found. Skipping checksum verification."
    print_warning "It is recommended to manually verify checksums."
fi

# Step 4: Extract archive
print_info "Step 4/4: Extracting dataset (this will take several minutes)..."
if cat contacts_assembly_json.tar.gz.part* | tar xzf -; then
    print_info "Extraction completed successfully"
else
    print_error "Extraction failed"
    exit 1
fi

# Verify extraction
if [ -d "$OUTPUT_DIR" ]; then
    FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
    print_info "Dataset extracted: $FILE_COUNT files in $OUTPUT_DIR/"
else
    print_error "Expected directory '$OUTPUT_DIR' not found after extraction"
    exit 1
fi

# Cleanup
if [ "$CLEANUP" = true ]; then
    print_info "Cleaning up archive parts..."
    rm -f contacts_assembly_json.tar.gz.part*
    rm -f "$CHECKSUM_FILE"
    print_info "Cleanup complete"
else
    print_info "Archive parts retained. Use --cleanup flag to remove them automatically."
fi

# Summary
print_info "===== COMPLETE ====="
print_info "Dataset successfully downloaded and extracted to: $OUTPUT_DIR/"
print_info "Total files: $FILE_COUNT"

if [ "$CLEANUP" = false ]; then
    ARCHIVE_SIZE=$(du -sh contacts_assembly_json.tar.gz.part* 2>/dev/null | tail -1 | awk '{print $1}')
    print_info "Archive parts (~93.2 GB) still present. To remove them:"
    echo "  rm -f contacts_assembly_json.tar.gz.part*"
    echo "  rm -f $CHECKSUM_FILE"
fi

DATASET_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
print_info "Dataset size: $DATASET_SIZE"
