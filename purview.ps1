#the modue must be installed first from AIP UL client (aka.ms/AIPClient)
#Get-Module -ListAvailable PurviewInformationProtection
#$env:PSModulePath += ";C:\Program Files (x86)\Microsoft Purview Information Protection\Powershell"


Import-Module PurviewInformationProtection


foreach ($file in (Get-ChildItem -Path "C:\Users\azghafar\OneDrive - Microsoft\Code\AI_Projects\src\playground\ODPP\data\input" -Filter *.pdf))
{
    Write-Host "File: $($file.FullName)"
    #Get-FileLabel -FilePath $file.FullName
    Remove-FileLabel $file.FullName -RemoveProtection
}


#$filepath = "C:\Users\azghafar\OneDrive - Microsoft\Code\AI_Projects\src\playground\ODPP\data\input\RD55U0100_EN_3209040_740786_4931 (1).pdf"
#$filepath = "C:\Users\azghafar\OneDrive - Microsoft\Code\AI_Projects\src\playground\ODPP\data\input\RD55U0100_EN_7948736_1554987_5851 - Copy (1).pdf"

#$filepath = "C:\Users\azghafar\OneDrive - Microsoft\Code\AI_Projects\src\playground\ODPP\data\input\RD55U0100_EN_10246554_2505342_4385.pdf"
#$filepath = "C:\Users\azghafar\OneDrive - Microsoft\Code\AI_Projects\src\playground\ODPP\data\input\RD55U0100_EN_10245066_2505065_4296 (1).pdf"


#Remove-FileLabel $filepath -RemoveLabel -JustificationMessage 'The previous label no longer applies'

#Remove-FileLabel $filepath -RemoveProtection


