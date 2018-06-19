# Backpropagation
Implementación de Backpropagation con librerias paralelizadas

## Especificaciones del Equipo

* Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz 64 bits
* 8GB RAM
* Ubuntu 16.04

### Librerias
* Armadillo [Pasos para instalar](https://solarianprogrammer.com/2017/03/24/getting-started-armadillo-cpp-linear-algebra-windows-mac-linux/)
Esta libreria nos permite paralalelizar las operaciones
### Para ejecutar:
Se uso codeblocks con -std=c++11 e instalación previa de Armadillo y  -lopenblas -llapack

## Implementación
Para la implementacion se usó diferentes clases:
* Neurona `neuron.cpp` `neuron.h`
* Capa `layer.cpp` `layer.h`
* Red  `network.cpp` `network.h`

Para crear una red `main.cpp`:
1. Definir numero de neuronas en cada capa oculta:
```c++
    vector<int> hidden; //Vector que incluye  el numero de neuronas por capa oculta
    hidden.push_back(8);hidden.push_back(6);
```
2. Crear red definiendo el numero de capas totales, número de parametros de la entrada, capas ocultas,numero de clases para la capa salida.
```c++
    Network * my_net = new Network(4,4,hidden,3); // 4 niveles(1 entrada, 2 hidden, 1 salida) ,entrada de 4 parametros, clases de salida
```

3. Cargar data y normalizar
```c++
    int Es=120;// cantidad train
    my_net->loadDataFlowers("irisTraining.txt", Es, inputs, outputs);
    my_net->normalize(IN, inputs);

```
4. Realizar el forward y el backpropagation para cada una de las entradas
```c++
for(int i=0 ;i<Es; i++) //Iterar dentro del conjunto de entrenamiento
        {
            double t=0.00001;
            my_net->init(IN[i],outputs[i], t);
            //cout<<"entrada:  "<< i << "   ****  era ***  "<<era<<endl;
            my_net->forward();
            delta=my_net->sumSquareError();
            //cout<<"SumsquareError de la capa:"<<delta<<endl;
            if(delta>0.000001)
                my_net->backpropagation();
            //my_net->forward();
            FinalErrors.push_back(delta);
            era++;
        }
```

## Capturas de la ejecución
Los experimentos se reralizarón con el Dataset de Iris, con 120 entradas para el Train y 30 para el Test

* Pesos despues del train y error obtenido

<img src="/images/pesos.png" alt="Screenshot1"/>

* Numero de aciertos el test

<img src="/images/test-results.png" alt="Screenshot2"/>

