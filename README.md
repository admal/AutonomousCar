## Projekt 2 - konspekt
#### Cel
Zaprojektowanie i implementacja sieci neuronowej potrafiącej bez udziału człowieka podróżować po generowanych torach w symulatorze i utrzymywać się na drodze. 

#### Technologie
* Python 3
* TensorFlow
* Udacity self driving car simulator ([repo](https://github.com/udacity/self-driving-car-sim))

#### Harmonogram
* 28.10 - rozpoczęcie researchu dostępnych technologii i prac naukowych dotyczących tego problemu
* od 04.11 - konfiguracja i poznanie symulatora *Udacity*, planowanie designu modelu sieci i aplikacji klienckiej
* od 07.11 - implementacja modelu, aplikacji klienckiej
* od 25.11 - uczenie sieci i pierwsze testy, poprawki do modelu
* od 03.12 - implementacja wizualizacji wyników testowych
* 23.01 - zakończenie projektu

#### Dane uczące
Dane generowane będą poprzez jazdę w symulatorze *Udacity*. Podczas jazdy tworzony jest plik _.csv_, który zawiera takie dane jak ścieżki fizyczne do screenshotów z jazdy (3 screenshoty na każdą klatkę), obrót kierownicy itd. Dane z tego pliku posłużą jako dane wejściowe do sieci neuronowej. Nauczony model będzie następnie użyty do prowadzenia samochodu w symulatorze bez pomocy człowieka.

#### Testowanie nauczonego modelu
Już nauczony model będzie testowany z użyciem symulatora *Udacity*. W tym celu potrzebne będzie stworzenie aplikacji klienckiej, która będzie komunikować się z symulatorem w celu pobierania informacji o aktualnym stanie pojazdu i wysyłania wyniku sieci do symulatora. Testowanie będzie polegac na porównywaniu czasu, przez który samochód był zdolny utrzymać się na drodze. Celem będzie osiągnięcie modelu, który na dowolnej konfiguracji toru będzie mógł swobodnie prowadzić i utrzymywać się na drodze.

#### Źródła
* https://www.youtube.com/watch?v=EaY5QiZwSP4
* https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
* https://github.com/udacity/self-driving-car-sim
