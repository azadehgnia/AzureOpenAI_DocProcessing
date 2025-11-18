#This script uses the Purview Information Protection PowerShell module to remove the sensitivity labels and protection from PDF files in a specified directory.
#The source code in this solution won't be able to run if the documents are protected.
#Removing the protection is not recommended in production scenarios, this is just for demo purposes.

#the modue must be installed first from AIP UL client (aka.ms/AIPClient)
#Get-Module -ListAvailable PurviewInformationProtection
#$env:PSModulePath += ";C:\Program Files (x86)\Microsoft Purview Information Protection\Powershell"


Import-Module PurviewInformationProtection


foreach ($file in (Get-ChildItem -Path "C:\Users\....\input" -Filter *.pdf))
{
    Write-Host "File: $($file.FullName)"
    #Get-FileLabel -FilePath $file.FullName
    Remove-FileLabel $file.FullName -RemoveProtection
}


