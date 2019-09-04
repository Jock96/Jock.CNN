﻿namespace CNN.Core.Layers
{
    using CNN.Core.Models;
    
    using System.Collections.Generic;
    using System;

    /// <summary>
    /// Класс скрытого слоя.
    /// </summary>
    public class HiddenLayer
    {
        /// <summary>
        /// Список нейронов скрытого слоя.
        /// </summary>
        private List<NeuronModel> _convolutionalLayer;

        /// <summary>
        /// Данные скрытого слоя.
        /// </summary>
        private List<NeuronModel> _hiddenLayerData;

        /// <summary>
        /// Класс скрытого слоя.
        /// </summary>
        /// <param name="convolutionalLayer">Данные свёрточного слоя.</param>
        public HiddenLayer(List<NeuronModel> convolutionalLayer)
        {
            _convolutionalLayer = convolutionalLayer;
        }

        /// <summary>
        /// Инициализация слоя.
        /// </summary>
        public void Initialize()
        {
            _hiddenLayerData = new List<NeuronModel>();

            var countOfNeuronsInHiddenLayer = (int)_convolutionalLayer.Count / 2;

            for (var index = 0; index < countOfNeuronsInHiddenLayer; ++index)
                _hiddenLayerData.Add(CreateNeuronByData());
        }

        /// <summary>
        /// Возвращает нейроны скрытого слоя.
        /// </summary>
        /// <returns></returns>
        public List<NeuronModel> GetLayerNeurons() => _hiddenLayerData;

        /// <summary>
        /// Создать нейрон по имеющимся данным.
        /// </summary>
        /// <returns>Возвращает нейрон.</returns>
        private NeuronModel CreateNeuronByData()
        {
            var inputs = new List<double>();
            var weights = new List<double>();

            foreach (var neuron in _convolutionalLayer)
            {
                inputs.Add(neuron.Output);
                weights.Add(GetInitializedWeight());
            }

            var hiddenLayerNeuron = new NeuronModel(inputs, weights);
            return hiddenLayerNeuron;
        }

        /// <summary>
        /// Инициализация случайных весов.
        /// </summary>
        /// <returns>Возвращает случайный вес.</returns>
        private double GetInitializedWeight() => new Random().NextDouble();
    }
}