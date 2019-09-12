namespace CNN.Core.Utils
{
    using CNN.BL.Constants;
    using CNN.BL.Enums;
    using CNN.BL.Helpers;

    using CNN.Core.Layers;
    using CNN.Core.Models;

    using System.Collections.Generic;
    using System.IO;

    /// <summary>
    /// Инструмент сохранения весов.
    /// </summary>
    public static class WeightSaveUtil
    {
        /// <summary>
        /// Сохранить как.
        /// </summary>
        /// <param name="path">Путь.</param>
        /// <param name="layers">Список слоёв.</param>
        public static void SaveAs(string path, List<Layer> layers)
        {
            if (string.IsNullOrEmpty(path))
                path = Path.Combine(PathHelper.GetResourcesPath(), 
                    FileConstants.WEIGHTS_DIRECTORY);

            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            var outputLayer = (OutputLayer)layers.Find(layer =>
                    layer.LayerType.Equals(LayerType.Output));

            OutputLayerSave(path, outputLayer);

            var hiddenLayer = (HiddenLayer)layers.Find(layer =>
            layer.LayerType.Equals(LayerType.Hidden));

            HiddenLayerSave(path, hiddenLayer);
            FilterCoreSave(path);
        }

        /// <summary>
        /// Сохранить ядро фильтра.
        /// </summary>
        /// <param name="path">Путь сохранения.</param>
        private static void FilterCoreSave(string path)
        {
            var filterCore = FilterCoreModel.GetCore;
            var directoryToSave = Path.Combine(path, $"{MatrixConstants.MATRIX_NAME}");

            if (!Directory.Exists(directoryToSave))
                Directory.CreateDirectory(directoryToSave);

            var fileToSave = Path.Combine(directoryToSave, $"{MatrixConstants.MATRIX_NAME}" +
                $"{FileConstants.TEXT_EXTENSION}");

            using (var stream = new StreamWriter(fileToSave))
            {
                for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                    for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                    {
                        stream.Write($"{MatrixConstants.POSITION_IN_X_AXIS}{xIndex}" +
                            $"{MatrixConstants.KEY_SEPARATOR}" +
                            $"{MatrixConstants.POSITION_IN_Y_AXIS}{yIndex}" +
                            $"{filterCore[xIndex, yIndex]} ");
                    }
            }
        }

        /// <summary>
        /// Сохранение скрытого слоя.
        /// </summary>
        /// <param name="path">Путь.</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        private static void HiddenLayerSave(string path, HiddenLayer hiddenLayer)
        {
            if (hiddenLayer != null)
            {
                var neurons = hiddenLayer.GetLayerNeurons();

                foreach (var neuron in neurons)
                {
                    var directoryToSave = Path.Combine(path, 
                        LayersConstants.HIDDEN_LAYER_NAME);

                    if (!Directory.Exists(directoryToSave))
                        Directory.CreateDirectory(directoryToSave);

                    var fileToSave = Path.Combine(directoryToSave,
                        $"{neurons.IndexOf(neuron)}{FileConstants.TEXT_EXTENSION}");

                    using (var stream = new StreamWriter(fileToSave))
                    {
                        neuron.Weights.ForEach(weight => stream.Write(weight + " "));
                    }
                }
            }
        }

        /// <summary>
        /// Сохранение выходного слоя.
        /// </summary>
        /// <param name="path">Путь.</param>
        /// <param name="outputLayer">Выходной слой.</param>
        private static void OutputLayerSave(string path, OutputLayer outputLayer)
        {
            if (outputLayer != null)
            {
                var neuron = outputLayer.GetOutputNeuron();
                var directoryToSave = Path.Combine(path, LayersConstants.OUTPUT_LAYER_NAME);

                if (!Directory.Exists(directoryToSave))
                    Directory.CreateDirectory(directoryToSave);

                var fileToSave = Path.Combine(directoryToSave, $"{0}" +
                    $"{FileConstants.TEXT_EXTENSION}");

                using (var stream = new StreamWriter(fileToSave))
                {
                    neuron.Weights.ForEach(weight => stream.Write(weight + " "));
                }
            }
        }
    }
}
