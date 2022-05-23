setlocal enabledelayedexpansion
@echo off
set root=C:\Users\Tre\anaconda3
call %root%\Scripts\activate.bat %root%

set date=2022-03-03
set sub=1
FOR /L %%v IN (0, 1, 23) Do (
	set "h=0%%v"
	call twarc2 csv json/ukraine_!date!_!h:~-2!.jsonl csv/ukraine_!date!_!h:~-2!.csv


)

pause
