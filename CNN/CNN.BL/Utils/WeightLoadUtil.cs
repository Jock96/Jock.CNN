namespace CNN.BL.Utils
{
    using CNN.BL.Constants;
    using CNN.BL.Enums;

    using System;
    using System.Text;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using CNN.BL.Helpers;

    /// <summary>
    /// Инструмент загрузки весов.
    /// </summary>
    public class WeightLoadUtil
    {
        /// <summary>
        /// Путь до файлов весов.
        /// </summary>
        private string _path;

        /// <summary>
        /// Словарь, где ключ - тип весов, значение - список весов.
        /// </summary>
        private Dictionary<WeightsType, List<string>> _weightTypeToPathDictionary;

        /// <summary>
        /// Инструмент загрузки весов.
        /// </summary>
        /// <param name="path">Путь до файлов весов.</param>
        public WeightLoadUtil(string path)
        {
            _path = path;
        }

        /// <summary>
        /// Загрузить веса.
        /// </summary>
        public void Load()
        {
            try
            {
                _weightTypeToPathDictionary = new Dictionary<WeightsType, List<string>>();

                var outputLayerFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, LayersConstants.OUTPUT_LAYER_NAME)).ToList();

                _weightTypeToPathDictionary.Add(WeightsType.Output, outputLayerFiles);

                var hiddenLayerFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, LayersConstants.HIDDEN_LAYER_NAME)).ToList();

                _weightTypeToPathDictionary.Add(WeightsType.Hidden, hiddenLayerFiles);

                var coreFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, MatrixConstants.MATRIX_NAME)).ToList();

                _weightTypeToPathDictionary.Add(WeightsType.Core, coreFiles);
            }
            catch
            {
                ErrorHelper.DirectoryError();
            }
        }

        /// <summary>
        /// Получить данные.
        /// </summary>
        /// <returns>Вовзращает словарь, где ключ - тип весов, значение - список весов.</returns>
        public Dictionary<WeightsType, Dictionary<int, List<double>>> GetData()
        {
            var weightTypeToDataDictionary = new Dictionary<WeightsType, Dictionary<int, List<double>>>();

            foreach (var keyValuePair in _weightTypeToPathDictionary)
            {
                if (keyValuePair.Key.Equals(WeightsType.Output) ||
                    keyValuePair.Key.Equals(WeightsType.Hidden))
                    GetWeights(ref weightTypeToDataDictionary, keyValuePair);
            }

            return weightTypeToDataDictionary;
        }

        /// <summary>
        /// Получить веса слоя.
        /// </summary>
        /// <param name="weightTypeToDataDictionary">Словарь для заполнения.</param>
        /// <param name="keyValuePair">Пара ключ - тип весов, значение - пути к файлам весов.</param>
        private void GetWeights(ref Dictionary<WeightsType, Dictionary<int, List<double>>> weightTypeToDataDictionary, KeyValuePair<WeightsType, List<string>> keyValuePair)
        {
            var weightsValue = new List<double>();
            var neuronIndexToWeightsDictionary = new Dictionary<int, List<double>>();

            foreach (var path in keyValuePair.Value)
            {
                using (var stream = File.OpenRead(path))
                {
                    var array = new byte[stream.Length];
                    stream.Read(array, 0, array.Length);

                    var valueString = Encoding.Default.GetString(array);
                    var indexOfSeparator = 0;

                    do
                    {
                        indexOfSeparator = valueString.IndexOf(" ");

                        if (indexOfSeparator == -1)
                            continue;

                        var value = valueString.Remove(indexOfSeparator);

                        if (!double.TryParse(value, out var convertedValue))
                            ErrorHelper.ParseError();

                        weightsValue.Add(convertedValue);
                        valueString = valueString.Remove(0, value.Length + 1);
                    } while (indexOfSeparator != -1);
                }

                var indexOfNeuronString = Path.GetFileNameWithoutExtension(path);

                if (!int.TryParse(indexOfNeuronString, out var indexOfNeuron))
                    ErrorHelper.ParseError();

                neuronIndexToWeightsDictionary.Add(indexOfNeuron, weightsValue);
            }

            weightTypeToDataDictionary.Add(keyValuePair.Key, neuronIndexToWeightsDictionary);
        }

        /// <summary>
        /// Обновить ядро фильтра.
        /// </summary>
        /// <returns>Возвращает новое ядро для обновления.</returns>
        public double[,] GetNewCore()
        {
            var newCore = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            if (!_weightTypeToPathDictionary.TryGetValue(WeightsType.Core, out var corePathList))
                ErrorHelper.GetDataError();

            var corePath = corePathList.FirstOrDefault();

            if (corePath == null)
                ErrorHelper.GetDataError();

            using (var stream = File.OpenRead(corePath))
            {
                var array = new byte[stream.Length];
                stream.Read(array, 0, array.Length);

                var valueString = Encoding.Default.GetString(array);

                for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                    for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                    {
                        var stringToCompare = $"{MatrixConstants.POSITION_IN_X_AXIS}{xIndex}" +
                            $"{MatrixConstants.KEY_SEPARATOR}" +
                            $"{MatrixConstants.POSITION_IN_Y_AXIS}{yIndex}";

                        var cuttedStringValue = valueString.Replace(stringToCompare, string.Empty);
                        var indexOfSeparator = cuttedStringValue.IndexOf(" ");

                        var value = string.Empty;

                        if (indexOfSeparator == -1)
                        {
                            value = cuttedStringValue;
                        }
                        else
                        {
                            value = cuttedStringValue.Remove(indexOfSeparator);
                        }

                        if (!double.TryParse(value, out var prepearedValue))
                            ErrorHelper.ParseError();

                        newCore[xIndex, yIndex] = prepearedValue;
                        valueString = cuttedStringValue.Remove(0, value.Length + 1);
                    }
            }

            return newCore;
        }
    }
}
