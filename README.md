# Сравнение реализаций распараллеленного алгоритма Дейкстры
В данной работе был произведен анализ различных средств для распараллеливания алгоритма Дейкстры \[[1]\].
Рассматриваются три варианта реализации:
1. Последовательный алгоритм;
2. "Наивный" параллельный алгоритм, использующий примитивы синхронизации потоков;
3. Параллельный алгоритм, предложенный в работах \[[2]\], \[[3]\].

Распараллеливание производится как на центральном процессоре (OpenMP, OpenCL), 
так и с использованием графического ускорителя (OpenACC, Nvidia CUDA, OpenCL). В 
качестве языка реализации был выбран С++ стандарта C++ 11, для сборки используется 
CMake версии не ниже 3.9. 

# Структура проекта
Для удобства, вся программа для тестирования собирается в один исполняемый файл, а исходные
коды для каждого из вариантов алгоритма и технологии помещены в отдельные файлы. Ключевые файлы:
1. [main.cpp] -- точка входа. Именно здесь измеряется времена выполнения каждого из тестов.
2. [sequential.cpp] -- файл с последовательной реализацией алгоритма Дийкстры.
3. [parallel_omp.cpp] -- реализация "наивного" параллельного алгоритма для CPU с использованием OpenMP.
4. [parallel_cl.cpp], [dijkstra.cl] -- реализация паралельного алгоритма для GPU и CPU с использованием OpenCL (если поддерживается устройствами).
5. [parallel_acc.cpp] -- реализация паралельного алгоритма для GPU с использованием OpenACC.
6. [dijkstra.cu] -- реализаця паралельного алгоритма для GPU с использованием Nvidia CUDA.

# Сборка
Чтобы собрать проект, необходимо сначала сгенерировать Makefile. Делается это следующим образом: 
1. `?> mkdir build`
2. `?> cd build`
3. `?> cmake ..`

После этого, достаточно выполнить команду `make` внутри каталога `build`,  или выполнить команду
```
?> cmake --build ./build
```
из корня репозитория. 

Для включения NVidia CUDA в проекте, необходимо установить параметр `ENABLE_CUDA` в 1 в файле [CMakeLists.txt]. 

Если что-то не собирается, можно попытаться выполнить команду `make` с параметром `VERBOSE=1` из каталога `build`, 
чтобы увидеть команды, которые `make` выполнит. 

# Список литературы:
1. Dijkstra's algoritm / wikipedia.com
2. Accelerating large graph algorithms on the GPU using CUDA / Parwan Harish and P.J. Narayanan
3. CUDA Solutions for the SSSP Problem / Pedro J. Martín, Roberto Torres, and Antonio Gavilanes

[1]: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
[2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.4206&rep=rep1&type=pdf
[3]: https://link.springer.com/chapter/10.1007/978-3-642-01970-8_91

[main.cpp]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/sequential.cpp
[sequential.cpp]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/sequential.cpp
[parallel_omp.cpp]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/parallel_omp.cpp
[parallel_cl.cpp]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/parallel_cl.cpp
[parallel_acc.cpp]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/parallel_acc.cpp
[dijkstra.cu]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/gpu/dijkstra.cu
[dijkstra.cl]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/src/gpu/dijkstra.cl
[CMakeLists.txt]: https://github.com/Morozov-5F/parallel-dijkstra-comparison/blob/master/CMakeLists.txt
