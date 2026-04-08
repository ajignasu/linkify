# Manual Download Instructions

### 1. Download All Parts

Using `wget`:
```bash
wget https://fusion-360-gallery-assembly-interfaces.s3.us-west-2.amazonaws.com/public-archives/contacts_assembly_json.tar.gz.part{aa,ab,ac,ad,ae,af,ag,ah,ai,aj}
wget https://fusion-360-gallery-assembly-interfaces.s3.us-west-2.amazonaws.com/public-archives/contacts_assembly_json.tar.gz.sha256
```

Using `curl`:
```bash
for part in aa ab ac ad ae af ag ah ai aj; do
  curl -O https://fusion-360-gallery-assembly-interfaces.s3.us-west-2.amazonaws.com/public-archives/contacts_assembly_json.tar.gz.part${part}
done
curl -O https://fusion-360-gallery-assembly-interfaces.s3.us-west-2.amazonaws.com/public-archives/contacts_assembly_json.tar.gz.sha256
```

### 2. Verify Checksums

On Linux/macOS:
```bash
shasum -c contacts_assembly_json.tar.gz.sha256
```

On Windows (PowerShell):
```powershell
Get-Content contacts_assembly_json.tar.gz.sha256 | ForEach-Object {
    $hash, $file = $_ -split '\s+', 2
    $computed = (Get-FileHash -Algorithm SHA256 $file.TrimStart('*')).Hash.ToLower()
    if ($computed -eq $hash) {
        Write-Host "OK: $file"
    } else {
        Write-Host "FAILED: $file"
    }
}
```

### 3. Extract the Archive

Reassemble and extract in one command:
```bash
cat contacts_assembly_json.tar.gz.part* | tar xzf -
```

Or extract to a specific directory:
```bash
cat contacts_assembly_json.tar.gz.part* | tar xzf - -C /path/to/destination
```

### 4. Clean Up (Optional)

After successful extraction, you can remove the archive parts:
```bash
rm contacts_assembly_json.tar.gz.part*
rm contacts_assembly_json.tar.gz.sha256
```
