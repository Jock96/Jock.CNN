namespace CNN.Core.Extensions
{
    using CNN.Core.Models;

    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Класс расширений для операций с моделями нейронов.
    /// </summary>
    public static class NeuronModelExtension
    {
        /// <summary>
        /// Обновление весов нейрона.
        /// </summary>
        /// <param name="neuronModel">Модель нейрона.</param>
        /// <param name="weights">Новые веса.</param>
        public static void UpdateWeights(this NeuronModel neuronModel, List<double> weights)
        {
            if (neuronModel.Weights.Count != weights.Count)
            {
                var exception = new Exception("Несоответствие по количеству весов.");

                Console.WriteLine(BL.Constants.ConsoleMessageConstants.ERROR_MESSAGE + 
                    exception.ToString());

                Console.ReadKey();
                return;
            }

            neuronModel.LastWeights = neuronModel.Weights;
            neuronModel.Weights = weights;
        }
    }
}
