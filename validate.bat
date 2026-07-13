@echo off
REM paper-validator quick launcher
REM Usage: validate [claim-name|all] [--trials N]
REM Importers: user CLI, any agent via Bash tool
REM Callers: main.py cmd_claim()
REM Schema: ClaimReport {claim_id, verdict, effect_size, metrics}
REM User verbatim: "create startup script to easily run paper-validator"

cd /d "%~dp0"

if defined DEEPSEEK_API_KEY goto :run
if defined OPENAI_API_KEY goto :run
for /f "tokens=*" %%i in ('python -c "import json,os;c=json.load(open(os.path.expanduser('~/.claude/settings.json')));print(c.get('env',{}).get('DEEPSEEK_API_KEY',''))" 2^>nul') do set DEEPSEEK_API_KEY=%%i

:run
python -m paper_validator claim --claim %* 2>&1
