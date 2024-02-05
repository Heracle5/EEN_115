% FILEPATH: /Users/hangjenchan/Desktop/Project/EEN_115/test.m
% This script loads topology matrices for Italy and Germany from text files,
% creates directional graphs using the loaded data, and plots the graphs
% with edge labels. It also saves the plotted graphs as PNG images in the
% specified directory.
clear all;
clc;
close all;

% Load topology matrices for Italy and Germany
topology_matrix_Italy=load('/Users/hangjenchan/Desktop/Project/EEN_115/Italian-10nodes/IT10-topology.txt');
topology_matrix_Germany=load('/Users/hangjenchan/Desktop/Project/EEN_115/Germany-7nodes/G7-topology.txt');

% Create tables for new edges with end nodes and weights
new_edges_Italy = table(topology_matrix_Italy(:,4:5),topology_matrix_Italy(:,6),'VariableNames',{'EndNodes','Weight'});
new_edges_Germany = table(topology_matrix_Germany(:,4:5),topology_matrix_Germany(:,6),'VariableNames',{'EndNodes','Weight'});

% Create directional graphs using the new edges tables
directional_graph_Italy=graph(new_edges_Italy);
directional_graph_Germany=graph(new_edges_Germany);

% Plot and save the graphs for Italy and Germany
for i=1:2
    figure(i);
    if i==1
        plot(directional_graph_Italy,'EdgeLabel',new_edges_Italy.Weight);
        % Save the Italy graph as a PNG image in the Graph folder
        saveas(figure(1),'/Users/hangjenchan/Desktop/Project/EEN_115/Graph/Italy_topo.png');
    else
        plot(directional_graph_Germany,'EdgeLabel',new_edges_Germany.Weight);
        % Save the Germany graph as a PNG image in the Graph folder
        saveas(figure(2),'/Users/hangjenchan/Desktop/Project/EEN_115/Graph/Germany_topo.png');
    end
end