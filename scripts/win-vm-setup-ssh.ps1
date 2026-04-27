# Run once inside the Windows VM (as Administrator) to enable SSH
# Then from the Linux host, run: scripts/test-win-vm.sh

# Install OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.0.5

# Start + enable SSH service
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# Allow SSH through firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (GigaVector test)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 `
    -ErrorAction SilentlyContinue

# Print your public key from the Linux host and paste below
# (or run: ssh-copy-id -p 2222 $env:USERNAME@localhost from the Linux host)
Write-Host ""
Write-Host "SSH enabled. Your Windows username: $env:USERNAME"
Write-Host "From Linux host run: ssh-copy-id -p 2222 -i ~/.ssh/id_ed25519.pub $env:USERNAME@localhost"
