package utils;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class EvaluationFile {

	private final String path;
	private final List<ClassifierEvaluation> evaluationsList;
	private final String description;

	public EvaluationFile(String path, List<ClassifierEvaluation> evaluationList, String description) {
		this.path = path;
		this.evaluationsList = evaluationList;
		this.description = description;
	}


	public void csvWrite(String projectName) throws IOException {
		try(FileWriter fileWriter = new FileWriter(this.path + "_" + this.description + ".csv")){

			fileWriter.append("DATASET, #TRAINING_RELEASES, %TRAINING_INSTANCES, CLASSIFIER, FEATURE_SELECTION, BALANCING, COST_SENSITIVE, PRECISION, RECALL, AUC, KAPPA, TP, FP, TN, FN\n");

			for (ClassifierEvaluation classifierEvaluation : this.evaluationsList) {

				fileWriter.append(projectName);

				if (this.description.equals("details")) {
					fileWriter.append(",");
					fileWriter.append(String.valueOf(classifierEvaluation.getWalkForwardIterationIndex()-1));
					fileWriter.append(",");
					fileWriter.append(String.valueOf(classifierEvaluation.getTrainingPercent()));
				} else {
					fileWriter.append(",None,None");
				}

				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getClassifier()));

				if (classifierEvaluation.isFeatureSelection()) {
					fileWriter.append(",Best First,");
				} else {
					fileWriter.append(",None,");
				}

				if (classifierEvaluation.isSampling()) {
					fileWriter.append("Undersampling,");
				} else {
					fileWriter.append("None,");
				}

				if (classifierEvaluation.isCostSensitive()) {
					fileWriter.append("Sensitive learning,");
				} else {
					fileWriter.append("None,");
				}

				fileWriter.append(String.valueOf(classifierEvaluation.getPrecision()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getRecall()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getAuc()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getKappa()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getTp()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getFp()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getTn()));
				fileWriter.append(",");
				fileWriter.append(String.valueOf(classifierEvaluation.getFn()));

				fileWriter.append("\n");
			}
		}
	}
}
