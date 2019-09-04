namespace CNN.Core.Layers
{
    using System;
    using System.Collections.Generic;
    using CNN.BL.Enums;
    using Models;

    /// <summary>
    /// Класс выходного слоя.
    /// </summary>
    public class OutputLayer : Layer
    {
        /// <summary>
        /// Значения скрытого слоя.
        /// </summary>
        private List<NeuronModel> _hiddenLayerData;

        /// <summary>
        /// Значение выходного нейрона.
        /// </summary>
        private NeuronModel _outputNeuron;

        /// <summary>
        /// Тип слоя.
        /// </summary>
        public override LayerType LayerType => LayerType.Output;

        public OutputLayer(List<NeuronModel> hiddenLayerData)
        {
            _hiddenLayerData = hiddenLayerData;
        }

        /// <summary>
        /// Инициализация слоя.
        /// </summary>
        public void Initilize()
        {
            var inputs = new List<double>();
            var weights = new List<double>();

            foreach (var neuron in _hiddenLayerData)
            {
                inputs.Add(neuron.Output);
                weights.Add(GetInitializedWeight());
            }

            _outputNeuron = new NeuronModel(inputs, weights);
        }

        /// <summary>
        /// Инициализация случайных весов.
        /// </summary>
        /// <returns>Возвращает случайный вес.</returns>
        private double GetInitializedWeight() => new Random().NextDouble();

        /// <summary>
        /// Получить значение выходного нейрона.
        /// </summary>
        /// <returns>Возвращает выходной нейрон.</returns>
        public NeuronModel GetOutputNeuron() => _outputNeuron;
    }
}
