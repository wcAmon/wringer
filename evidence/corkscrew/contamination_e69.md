# E69 污染檢查(calib_e58r1b / bleachers-e58r2 vs IFEval / HumanEval / GSM8K)

詞級 token(\w+ 或單標點,小寫);hit13 = 含任一 13-gram 出現於語料;run≥N = 最長連續共同串 ≥ N token。

- 語料 `bank`:11192 文件,3,114,257 詞,2,607,662 個 8-gram;HumanEval entry_point `def name(` 撞名 18/164
- 語料 `calib`:384 文件,4,569,921 詞,3,151,244 個 8-gram;HumanEval entry_point `def name(` 撞名 4/164

| 語料 | 測試欄位 | n | hit13 | hit13% | run≥13 | run≥25 | run≥50 | run≥100 | max_run | mean_run |
|---|---|---|---|---|---|---|---|---|---|---|
| bank | ifeval:prompt | 541 | 4 | 0.7 | 4 | 0 | 0 | 0 | 14 | 2.33 |
| bank | humaneval:prompt | 164 | 55 | 33.5 | 55 | 2 | 0 | 0 | 41 | 10.03 |
| bank | humaneval:canonical_solution | 164 | 31 | 18.9 | 31 | 4 | 0 | 0 | 37 | 7.04 |
| bank | humaneval:test | 164 | 24 | 14.6 | 24 | 7 | 0 | 0 | 42 | 6.02 |
| bank | gsm8k:question | 1319 | 0 | 0.0 | 0 | 0 | 0 | 0 | 12 | 0.44 |
| bank | gsm8k:answer | 1319 | 1 | 0.1 | 1 | 0 | 0 | 0 | 14 | 0.53 |
| calib | ifeval:prompt | 541 | 1 | 0.2 | 1 | 0 | 0 | 0 | 13 | 1.43 |
| calib | humaneval:prompt | 164 | 33 | 20.1 | 33 | 1 | 0 | 0 | 25 | 8.23 |
| calib | humaneval:canonical_solution | 164 | 19 | 11.6 | 19 | 1 | 0 | 0 | 29 | 4.91 |
| calib | humaneval:test | 164 | 13 | 7.9 | 13 | 3 | 0 | 0 | 42 | 5.24 |
| calib | gsm8k:question | 1319 | 0 | 0.0 | 0 | 0 | 0 | 0 | 10 | 0.27 |
| calib | gsm8k:answer | 1319 | 1 | 0.1 | 1 | 0 | 0 | 0 | 13 | 0.2 |

## 命中樣例(run ≥ 13,每欄最多 10)
### bank|ifeval:prompt
- id=2704 run=14 src=bank:ifshape:ifshape-002083 :: `write a haiku about foolish behavior in the form of a question , for an audience of young readers . it should include the topic of not studying . give exactly t`
- id=2683 run=13 src=bank:ifshape:ifshape-002083 :: `i ' ve got a collection of military insignia that i ' d like to get rid of , but i don ' t know how . can you help me ? give exactly two different responses , s`
- id=322 run=13 src=bank:ifshape:ifshape-002083 :: `who is joe biden and donald trump ' s national security advisors ? responses should be separated by 6 asterisk symbols ( * * * * * * ) . in other words , your o`
- id=3324 run=13 src=bank:ifshape:ifshape-002083 :: `write two versions of itinerary for a 7 day trip to hawaii , designed for college students . separate the two versions with 6 asterisk symbols ( * * * * * * ) .`
### bank|humaneval:prompt
- id=HumanEval/115 run=41 src=bank:code:code-001491 :: `def max_fill ( grid , capacity ) : import math " " " you are given a rectangular grid of wells . each row represents a single well , and each 1 in a row represe`
- id=HumanEval/129 run=28 src=bank:code:code-000037 :: `def minpath ( grid , k ) : " " " given a grid with n rows and n columns ( n > = 2 ) and a positive integer k , each cell of the grid contains a value`
- id=HumanEval/74 run=24 src=bank:code:code-002155 :: `def total_match ( lst1 , lst2 ) : ' ' ' write a function that accepts two lists of strings and returns the list that has total number of chars in the all string`
- id=HumanEval/116 run=22 src=bank:code:code-000019 :: `def sort_array ( arr ) : " " " in this kata , you have to sort an array of non - negative integers according to number of ones in their binary representation in`
- id=HumanEval/69 run=21 src=bank:code:code-000045 :: `def search ( lst ) : ' ' ' you are given a non - empty list of positive integers . return the greatest integer that is greater than zero , and has a frequency g`
- id=HumanEval/110 run=21 src=bank:code:code-000004 :: `def exchange ( lst1 , lst2 ) : " " " in this problem , you will implement a function that takes two lists of numbers , and determines whether it is possible to `
- id=HumanEval/26 run=20 src=bank:code:code-000004 :: `from typing import list def remove_duplicates ( numbers : list [ int ] ) - > list [ int ] : " " " from a list of integers , remove all elements that occur more `
- id=HumanEval/64 run=20 src=bank:code:code-000574 :: `fix = " " " add more test cases . " " " def vowels_count ( s ) : " " " write a function vowels_count which takes a string representing a word as input and retur`
- id=HumanEval/9 run=19 src=bank:code:code-000497 :: `from typing import list , tuple def rolling_max ( numbers : list [ int ] ) - > list [ int ] : " " " from a given list of integers , generate a list of rolling m`
- id=HumanEval/33 run=19 src=bank:code:code-000029 :: `def sort_third ( l : list ) : " " " this function takes a list l and returns a list l ' such that l ' is identical to l in the indicies that are not divisible b`
### bank|humaneval:canonical_solution
- id=HumanEval/147 run=37 src=bank:code:code-000010 :: `a = [ i * i - i + 1 for i in range ( 1 , n + 1 ) ] ans = [ ] for i in range ( n ) : for j in range ( i`
- id=HumanEval/94 run=32 src=bank:code:code-000006 :: `def isprime ( n ) : for i in range ( 2 , int ( n * * 0 . 5 ) + 1 ) : if n % i = = 0 : return false return true maxx =`
- id=HumanEval/129 run=30 src=bank:code:code-001933 :: `n = len ( grid ) val = n * n + 1 for i in range ( n ) : for j in range ( n ) : if grid [ i ] [ j ] = = 1`
- id=HumanEval/80 run=27 src=bank:code:code-001095 :: `if len ( s ) < 3 : return false for i in range ( len ( s ) - 2 ) : if s [ i ] = = s [ i + 1 ] or s [ i`
- id=HumanEval/71 run=23 src=bank:code:code-001428 :: `if a + b < = c or a + c < = b or b + c < = a : return - 1 s = ( a + b + c ) / 2 area = ( s`
- id=HumanEval/106 run=23 src=bank:code:code-000010 :: `ret = [ ] for i in range ( 1 , n + 1 ) : if i % 2 = = 0 : x = 1 for j in range ( 1 , i + 1 ) : x`
- id=HumanEval/107 run=21 src=bank:code:code-000042 :: `def is_palindrome ( n ) : return str ( n ) = = str ( n ) [ : : - 1 ] even_palindrome_count = 0 odd_palindrome_count = 0 for i in range ( 1 , n + 1 )`
- id=HumanEval/130 run=20 src=bank:code:code-000366 :: `if n = = 0 : return [ 1 ] my_tri = [ 1 , 3 ] for i in range ( 2 , n + 1 ) : if i % 2 = = 0 : my_tri . append`
- id=HumanEval/11 run=17 src=bank:code:code-000405 :: `def xor ( i , j ) : if i = = j : return ' 0 ' else : return ' 1 ' return ' ' . join ( xor ( x , y ) for x , y`
- id=HumanEval/55 run=17 src=bank:code:code-000027 :: `if n = = 0 : return 0 if n = = 1 : return 1 return fib ( n - 1 ) + fib ( n - 2 )`
### bank|humaneval:test
- id=HumanEval/115 run=42 src=bank:code:code-001491 :: `def check ( candidate ) : # check some simple cases assert true , " this prints if this assert fails 1 ( good for debugging ! ) " assert candidate ( [ [ 0 , 0 ,`
- id=HumanEval/129 run=42 src=bank:code:code-000008 :: `def check ( candidate ) : # check some simple cases print assert candidate ( [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ]`
- id=HumanEval/96 run=41 src=bank:code:code-003232 :: `def check ( candidate ) : assert candidate ( 5 ) = = [ 2 , 3 ] assert candidate ( 6 ) = = [ 2 , 3 , 5 ] assert candidate ( 7 ) = = [`
- id=HumanEval/87 run=39 src=bank:code:code-000670 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ [ 1 , 2 , 3 , 4 , 5 , 6 ] , [ 1 , 2 , 3 , 4 , 1 ,`
- id=HumanEval/142 run=30 src=bank:code:code-000207 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 ] ) = = 6 assert candidate ( [ 1 , 4 , 9 ] ) = = 14 assert`
- id=HumanEval/152 run=29 src=bank:code:code-000231 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 , 4 , 5 , 1 ] , [ 1 , 2 , 3 , 4 , 2 , -`
- id=HumanEval/111 run=26 src=bank:code:code-000059 :: `def check ( candidate ) : # check some simple cases assert candidate ( ' a b b a ' ) = = { ' a ' : 2 , ' b ' : 2 } , " this prints`
- id=HumanEval/74 run=24 src=bank:code:code-002155 :: `def check ( candidate ) : # check some simple cases assert true , " this prints if this assert fails 1 ( good for debugging ! ) " assert candidate ( [ ] , [ ] )`
- id=HumanEval/110 run=21 src=bank:code:code-000004 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 , 4 ] , [ 1 , 2 , 3 , 4 ] ) = = " yes " assert`
- id=HumanEval/126 run=17 src=bank:code:code-000004 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 5 ] ) = = true assert candidate ( [ 1 , 2 , 3 , 4 , 5 ] ) = = true assert`
### bank|gsm8k:answer
- id=410 run=14 src=bank:math:math-004221 :: `taffy is buying 1 pound for $ 3 and gets 1 pound half off . so 1 / 2 off of 1 pound is $ 3 / 2 = $ 1 . 50 + $ 3 . 00 = $`
### calib|ifeval:prompt
- id=13 run=13 src=calib:13 :: `what is the history of nyc prospect park ? please wrap your entire answer in json format . you can use markdown ticks such as ` ` ` . for example : ` ` ` json {`
### calib|humaneval:prompt
- id=HumanEval/129 run=25 src=calib:7 :: `def minpath ( grid , k ) : " " " given a grid with n rows and n columns ( n > = 2 ) and a positive integer k , each cell of the grid contains a value`
- id=HumanEval/74 run=23 src=calib:295 :: `def total_match ( lst1 , lst2 ) : ' ' ' write a function that accepts two lists of strings and returns the list that has total number of chars in the all string`
- id=HumanEval/78 run=21 src=calib:231 :: `def hex_key ( num ) : " " " you have been tasked to write a function that receives a hexadecimal number as a string and counts the number of hexadecimal digits `
- id=HumanEval/33 run=19 src=calib:47 :: `def sort_third ( l : list ) : " " " this function takes a list l and returns a list l ' such that l ' is identical to l in the indicies that are not divisible b`
- id=HumanEval/37 run=19 src=calib:47 :: `def sort_even ( l : list ) : " " " this function takes a list l and returns a list l ' such that l ' is identical to l in the odd indicies , while its values at`
- id=HumanEval/64 run=19 src=calib:312 :: `fix = " " " add more test cases . " " " def vowels_count ( s ) : " " " write a function vowels_count which takes a string representing a word as input and retur`
- id=HumanEval/107 run=19 src=calib:113 :: `def even_odd_palindrome ( n ) : " " " given a positive integer n , return a tuple that has the number of even and odd integer palindromes that fall within the r`
- id=HumanEval/111 run=19 src=calib:105 :: `def histogram ( test ) : " " " given a string representing a space separated lowercase letters , return a dictionary of the letter with the most repetition and `
- id=HumanEval/115 run=19 src=calib:192 :: `def max_fill ( grid , capacity ) : import math " " " you are given a rectangular grid of wells . each row represents a single well , and each 1 in a row represe`
- id=HumanEval/9 run=18 src=calib:255 :: `from typing import list , tuple def rolling_max ( numbers : list [ int ] ) - > list [ int ] : " " " from a given list of integers , generate a list of rolling m`
### calib|humaneval:canonical_solution
- id=HumanEval/80 run=29 src=calib:280 :: `if len ( s ) < 3 : return false for i in range ( len ( s ) - 2 ) : if s [ i ] = = s [ i + 1 ] or s [ i`
- id=HumanEval/147 run=23 src=calib:77 :: `a = [ i * i - i + 1 for i in range ( 1 , n + 1 ) ] ans = [ ] for i in range ( n ) : for j in range ( i`
- id=HumanEval/107 run=22 src=calib:1 :: `def is_palindrome ( n ) : return str ( n ) = = str ( n ) [ : : - 1 ] even_palindrome_count = 0 odd_palindrome_count = 0 for i in range ( 1 , n + 1 )`
- id=HumanEval/106 run=21 src=calib:7 :: `ret = [ ] for i in range ( 1 , n + 1 ) : if i % 2 = = 0 : x = 1 for j in range ( 1 , i + 1 ) : x`
- id=HumanEval/130 run=20 src=calib:115 :: `if n = = 0 : return [ 1 ] my_tri = [ 1 , 3 ] for i in range ( 2 , n + 1 ) : if i % 2 = = 0 : my_tri . append`
- id=HumanEval/11 run=17 src=calib:259 :: `def xor ( i , j ) : if i = = j : return ' 0 ' else : return ' 1 ' return ' ' . join ( xor ( x , y ) for x , y`
- id=HumanEval/55 run=16 src=calib:119 :: `if n = = 0 : return 0 if n = = 1 : return 1 return fib ( n - 1 ) + fib ( n - 2 )`
- id=HumanEval/63 run=16 src=calib:119 :: `if n = = 0 : return 0 if n = = 1 : return 0 if n = = 2 : return 1 return fibfib ( n - 1 ) + fibfib ( n - 2 ) + fibfib`
- id=HumanEval/129 run=16 src=calib:32 :: `n = len ( grid ) val = n * n + 1 for i in range ( n ) : for j in range ( n ) : if grid [ i ] [ j ] = = 1`
- id=HumanEval/96 run=15 src=calib:121 :: `primes = [ ] for i in range ( 2 , n ) : is_prime = true for j in range ( 2 , i ) : if i % j = = 0 : is_prime = false break if`
### calib|humaneval:test
- id=HumanEval/129 run=42 src=calib:7 :: `def check ( candidate ) : # check some simple cases print assert candidate ( [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ]`
- id=HumanEval/87 run=37 src=calib:106 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ [ 1 , 2 , 3 , 4 , 5 , 6 ] , [ 1 , 2 , 3 , 4 , 1 ,`
- id=HumanEval/152 run=27 src=calib:102 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 , 4 , 5 , 1 ] , [ 1 , 2 , 3 , 4 , 2 , -`
- id=HumanEval/74 run=23 src=calib:295 :: `def check ( candidate ) : # check some simple cases assert true , " this prints if this assert fails 1 ( good for debugging ! ) " assert candidate ( [ ] , [ ] )`
- id=HumanEval/145 run=21 src=calib:27 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 11 , - 1 , - 11 , - 12 ] ) = = [ - 1 , - 11 , 1 ,`
- id=HumanEval/142 run=20 src=calib:192 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 ] ) = = 6 assert candidate ( [ 1 , 4 , 9 ] ) = = 14 assert`
- id=HumanEval/115 run=19 src=calib:192 :: `def check ( candidate ) : # check some simple cases assert true , " this prints if this assert fails 1 ( good for debugging ! ) " assert candidate ( [ [ 0 , 0 ,`
- id=HumanEval/70 run=18 src=calib:27 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 1 , 2 , 3 , 4 ] ) = = [ 1 , 4 , 2 , 3 ] assert candidate ( [ 5`
- id=HumanEval/111 run=18 src=calib:105 :: `def check ( candidate ) : # check some simple cases assert candidate ( ' a b b a ' ) = = { ' a ' : 2 , ' b ' : 2 } , " this prints`
- id=HumanEval/126 run=16 src=calib:27 :: `def check ( candidate ) : # check some simple cases assert candidate ( [ 5 ] ) = = true assert candidate ( [ 1 , 2 , 3 , 4 , 5 ] ) = = true assert`
### calib|gsm8k:answer
- id=410 run=13 src=calib:34 :: `taffy is buying 1 pound for $ 3 and gets 1 pound half off . so 1 / 2 off of 1 pound is $ 3 / 2 = $ 1 . 50 + $ 3 . 00 = $`

## 裁讀(2026-09-11 22:4x)

- **GSM8K 乾淨**:question 零 13-gram 命中;answer 僅 1/1319(id 410,14 token,bank math-004221 同型題的算式片段)。
- **IFEval 乾淨**:命中 4(bank)/1(calib),全為指令模板句(「separate the two versions with 6 asterisk symbols」「give exactly two different responses」),非題目內容;最長 14 token。
- **HumanEval 有污染**:bleachers 題庫由 Max 老師自出題,近逐字重現了部分 HumanEval 題(含 docstring、canonical solution 與 `check(candidate)` 斷言資料);蓄水語料 calib_e58r1b(由題庫提示生成的軌跡)繼承其中一部分。
  - hit13 的高比例(prompt 34% / 20%)多為型別簽名 + docstring 樣板(`from typing import list def f ( numbers : list [ int ] ) - > list [ int ] : " " "`),不作污染證據;以最長連續共同串 run ≥ 25 為污染判準。
  - **S25(run ≥ 25,任一欄)= 10 題**:[80, 87, 94, 96, 111, 115, 129, 142, 147, 152](bank);calib 繼承 4 題(80/87/129/152)。
  - S20(run ≥ 20)= 22 題:[26, 64, 69, 71, 74, 78, 80, 87, 94, 96, 106, 107, 110, 111, 115, 116, 129, 130, 142, 145, 147, 152]。
  - entry_point 撞名 `def name(`:bank 18/164、calib 4/164(語意撞題參考,含常見名如 fib)。

### 去污染子集 HumanEval 分數(逐題 scores.jsonl 重算)

| 模型 | he 全 164 | 去 S25(154) | 去 S20(144) | S25 上通過率 | S20 上通過率 |
|---|---|---|---|---|---|
| bf16_w128(官方錨,verdict ANCH) | 94.51 | 94.16 | 94.37 | 100.0 | 95.5 |
| bf16(舊跑) | 95.73 | 96.10 | 97.18 | 90.0 | 86.4 |
| e69_p3b | 70.73 | 72.08 | 73.24 | 50.0 | 54.5 |
| e69_p3b_resA | 79.88 | 81.17 | 82.39 | 60.0 | 63.6 |
| e69_p3b_w | 84.15 | 84.42 | 84.51 | 80.0 | 81.8 |
| e69_p3b_w2 | 89.02 | 89.61 | 90.14 | 80.0 | 81.8 |
| e69_p3a | 70.73 | 71.43 | 71.83 | 60.0 | 63.6 |
| e69_p3a_resA | 86.59 | 86.36 | 85.92 | 90.0 | 90.9 |
| e69_p3a_w | 84.15 | 83.77 | 84.51 | 90.0 | 81.8 |
| e68_p0gptq | 86.59 | 85.71 | 85.92 | 100.0 | 90.9 |
| e68_p0rtn | 78.05 | 78.57 | 79.58 | 70.0 | 68.2 |
| e68_p1alpha | 90.85 | 90.91 | 91.55 | 90.0 | 86.4 |
| e68_p1gptq | 90.24 | 90.26 | 90.14 | 90.0 | 90.9 |
| e68_p2champ | 58.54 | 58.44 | 60.56 | 60.0 | 45.5 |

- p3b_w 去 S25 為 84.42(全 164 為 84.15)、去 S20 為 84.51:**旗標題不抬高 p3b_w 分數**,主結論 comp 0.9365 不受影響(he 項變動 < 0.3 pp)。
- 但蓄水增益在旗標題上(p3b→p3b_w:S25 50→80、S20 54.5→81.8)大於非旗標題(72.1→84.4),n=10/22 不足以定論,文章須照實揭露。
- 對照:P1-gptq(未蓄水,同語料校準)S25 90.0 與 bf16 同,顯示旗標題本身偏易。

### 發布動作
1. HF 文章與模型卡附本報告;HumanEval 同時報全集與去 S25 子集(p3b_w 84.15 / 84.42;bf16 錨 94.51 / 94.16;保留率 0.890 → 0.897)。
2. 第二模型(ROADMAP ⑥)與任何新蓄水前,先對題庫做去污染(以本腳本 run ≥ 20 剔除 bank 來源項:code-000006/000008/000010/000059/000207/000231/000670/001095/001491/003232 等),重生成 calib。
3. 進行中的 W2 鏈(p3b_w2/p3a_w)沿用 calib_e58r1b,不中途換料;其結果同樣附去 S25 子集分數。
