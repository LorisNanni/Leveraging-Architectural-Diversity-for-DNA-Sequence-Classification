% Legge il file CSV (specificando che la prima riga contiene le intestazioni)
data = readtable('C:\data\data\VMiner\OtherDatasets\ecoli_independent_test.csv');

% Estrarre l'ultima colonna usando l'indice dell'ultima colonna
labelsEcoli = data{:, end};

% Legge il file CSV (specificando che la prima riga contiene le intestazioni)
data = readtable('C:\data\data\VMiner\OtherDatasets\bert_enhancer_ind_test_set.csv');

% Estrarre l'ultima colonna usando l'indice dell'ultima colonna
labelsENH = data{:, end};
