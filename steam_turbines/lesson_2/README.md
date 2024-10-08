# Домашние задачи

* Задача 1

Построить `график` зависимости термического КПД от давления промежуточного перегрева водяного пара для условий примера: $Р_0$=12 МПа; $t_0$=530 °С; $Р_к$=5 кПа. Параметры промперегрева: $Р_{п.п.}$=1, 2, 3, 4, 5, 6 МПа; $t_{п.п.}$=530 °С.


* Задача 2

Построить график изолиний термического КПД от давления промежуточного перегрева и начальной температуры для Р_0$=12 МПа; $t_0$=530 °С; $Р_к$=5 кПа.. Графиком покрыть множество максимумов КПД. Для отрисовки прочитать [документацию](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)


* Задача 3

При удельных расходах водяного пара $d_{01}$= 2,9 кг/(кВт·ч) и $d_{02}$= 3,4 кг/(кВт·ч) оценить удельные расходы теплоты на выработку электроэнергии, приняв разность энтальпий $h_0$ – $h_{п.в.}$ = 2400 кДж/кг.


* Задача 4

Написать код для решения задачи оптимизации параметров промежуточного перегрева $P_{п.п.}$ и $t_{п.п.}$ для свободных начальных параметров $Р_0$, $t_0$, $Р_к$. Сделаем допущение, что начальная точка процесса расширения всегда находится в зоне перегретого пара. Решение должно выдавать параметры промежуточного перегрева и термический КПД при них. На основе этого кода собрать информацию об оптимальных давлениях промежуточного перегрева и температуры при $P_0$ от 5 до 12 МПа с шагом 1 МПа и температурой $t_0$ = 500 °С, $Р_к$=5 кПа. Сделать график функции оптимального давления промежуточного перегрева от давления $P_0$
