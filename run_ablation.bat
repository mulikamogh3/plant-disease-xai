@echo off
REM Ablation (Table B) on ResNet-50. Run after run_main.bat has finished.
set EPOCHS=30

python train.py --model resnet50 --mode bgrand  --epochs %EPOCHS% --resume || goto :err
python train.py --model resnet50 --mode attloss --epochs %EPOCHS% --resume || goto :err

python evaluate.py --run runs\resnet50_bgrand  || goto :err
python evaluate.py --run runs\resnet50_attloss || goto :err
python make_figures.py || goto :err
echo ABLATION DONE.
goto :eof

:err
echo A step failed. Fix the error above, then run this file again.
exit /b 1
