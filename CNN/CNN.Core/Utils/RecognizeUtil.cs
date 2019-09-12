namespace CNN.Core.Utils
{
    using CNN.Core.Layers;
    using CNN.Core.Models;
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Инструмент распознавания изображений.
    /// </summary>
    public class RecognizeUtil
    {
        /// <summary>
        /// Данныеи зображения.
        /// </summary>
        private double[,] _imageData;

        /// <summary>
        /// Словарь весов выходного слоя, где ключ - индекс нейрона, значение - список весов.
        /// </summary>
        private Dictionary<int, List<double>> _weightsInOutputLayer;

        /// <summary>
        /// Словарь весов скрытого слоя, где ключ - индекс нейрона, значение - список весов.
        /// </summary>
        private Dictionary<int, List<double>> _weightsInHiddenLayer;

        /// <summary>
        /// Инструмент распознавания изображений.
        /// </summary>
        /// <param name="weightsInOutputLayer">Веса выходного слоя.</param>
        /// <param name="weightsInHiddenLayer">Веса скрытого слоя.</param>
        /// <param name="imageData">Данные с изрбражения.</param>
        public RecognizeUtil(Dictionary<int, List<double>> weightsInOutputLayer,
            Dictionary<int, List<double>> weightsInHiddenLayer, double[,] imageData)
        {
            _weightsInHiddenLayer = weightsInHiddenLayer;
            _weightsInOutputLayer = weightsInOutputLayer;

            _imageData = imageData;
        }

        /// <summary>
        /// Инициализировать нейронную сеть.
        /// </summary>
        public void InitializeNetwork()
        {
            var inputLayer = new InputLayer(_imageData);
            inputLayer.Initialize();

            var convolutionalLayer = new ConvolutionalLayer(inputLayer.GetLayerNeurons());
            convolutionalLayer.RecognizeMode(FilterCoreModel.GetCore);

            var hiddenLayer = new HiddenLayer(convolutionalLayer.GetLayerNeurons());
            hiddenLayer.RecognizeMode(_weightsInHiddenLayer);

            var outputLayer = new OutputLayer(hiddenLayer.GetLayerNeurons());
            outputLayer.RecognizeMode(_weightsInOutputLayer);

            ToRecognizeImage(outputLayer.GetOutputNeuron().Output);
        }

        /// <summary>
        /// Распознать изображение.
        /// </summary>
        /// <param name="output">Вывод выходного слоя.</param>
        private void ToRecognizeImage(double output)
        {
            var message = string.Empty;

            if (output > 0.75 && output < 0.9)
            {
                Console.BackgroundColor = ConsoleColor.Yellow;
                message = "Вероятно на изображении цифра 1.";
            }

            if (output > 0.9)
            {
                Console.BackgroundColor = ConsoleColor.Cyan;
                message = "На изображении цифра 1.";
            }

            if (output < 0.75)
            {
                Console.BackgroundColor = ConsoleColor.Red;
                message = "Не удалось распознать изображение.";
            }

            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine(message);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.BackgroundColor = ConsoleColor.Black;
        }
    }
}
