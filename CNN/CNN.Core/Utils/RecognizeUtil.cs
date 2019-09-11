namespace CNN.Core.Utils
{
    using CNN.BL.Enums;

    using CNN.Core.Layers;

    using System.Collections.Generic;

    /// <summary>
    /// Инструмент распознавания изображений.
    /// </summary>
    public class RecognizeUtil
    {
        /// <summary>
        /// Словарь типов весов и значений.
        /// </summary>
        private Dictionary<WeightsType, List<double>> _weightsTypeToValuesDictionary;

        /// <summary>
        /// Данныеи зображения.
        /// </summary>
        private double[,] _imageData;

        /// <summary>
        /// Инструмент распознавания изображений.
        /// </summary>
        /// <param name="weightsTypeToValuesDictionary">Словарь типов весов и значений.</param>
        /// <param name="imageData">Данные изображения.</param>
        public RecognizeUtil(Dictionary<WeightsType, 
            List<double>> weightsTypeToValuesDictionary, double[,] imageData)
        {
            _weightsTypeToValuesDictionary = weightsTypeToValuesDictionary;
            _imageData = imageData;
        }

        public void Start()
        {
            var inputLayer = new InputLayer(_imageData);
            inputLayer.Initialize();
        }
    }
}
