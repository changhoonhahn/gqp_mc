# running `mini_mocha.py`

```bash
python mini_mocha.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${4} ${4} ${4}
```
1. data type: 'photo', 'spec', or 'specphoto'
2. simulation: 'lgal' or 'tng'
3. first galaxy index 
4. last galaxy index
5. noise model for the data
    - use 'legacy' for photo 
    - 'bgs0', 'bgs1', ... for spec
    - 'bgs0_legacy' for spec photo
6. fitting method: 'ispeculator'
7. model name: 'emulator' 
8. number of threads 
9. number of walkers
10. number of iterations for burn in
11. number of iterations: either specify a number of put `adaptive`
12. overwrite? If `True` existing file will be overwritten
13. postprocess SFR calculation from the MC chains


**examples**:
```bash
python -W ignore mini_mocha.py photo lgal 0 10 legacy ispeculator emulator 1 20 20 40 True True 
python -W ignore mini_mocha.py specphoto lgal 0 10 bgs0_legacy ispeculator emulator 1 20 20 40 True True 
```
