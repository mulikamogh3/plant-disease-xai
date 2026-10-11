@echo off
REM Main results: 3 models x (baseline, BG-Suppress), 30 epochs each.
REM Re-run this file after any crash: finished epochs/runs are resumed via --resume.
set EPOCHS=30

python train.py --model efficientnet_b0 --mode baseline   --epochs %EPOCHS% --resume || goto :err
python train.py --model resnet50        --mode baseline   --epochs %EPOCHS% --resume || goto :err
python train.py --model vit_small       --mode baseline   --epochs %EPOCHS% --resume || goto :err

python train.py --model efficientnet_b0 --mode bgsuppress --epochs %EPOCHS% --resume || goto :err
python train.py --model resnet50        --mode bgsuppress --epochs %EPOCHS% --resume || goto :err
python train.py --model vit_small       --mode bgsuppress --epochs %EPOCHS% --resume || goto :err

python evaluate.py --all || goto :err
python make_figures.py || goto :err
echo MAIN RUNS DONE. See results\results.csv and results\*.png
goto :eof

:err
echo A step failed. Fix the error above, then run this file again.
exit /b 1
