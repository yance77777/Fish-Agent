#define MyAppName "Fish-Agent"
#define MyAppVersion "3.4.0"
#define MyAppPublisher "Fish-Agent Team"
#define MyAppURL "https://github.com/yance77777/Fish-Agent"
#define MyAppExeName "Fish-Agent.exe"

[Setup]
AppId={{8A7F6D5E-4C3B-2A1F-9E8D-7C6B5A4F3E2D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\Fish-Agent
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=auto
DisableDirPage=no
OutputDir=..
OutputBaseFilename=Fish-Agent-V3.4.0-installer
SetupIconFile=assets\fish_agent_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
WizardStyle=modern
ChangesAssociations=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
Source: "dist\Fish-Agent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Fish-Agent"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\GitHub Project"; Filename: "{#MyAppURL}"
Name: "{autodesktop}\Fish-Agent"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Fish-Agent"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\Temp\Fish-Agent"
