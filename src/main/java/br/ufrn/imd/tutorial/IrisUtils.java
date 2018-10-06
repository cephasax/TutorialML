package br.ufrn.imd.tutorial;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * @author Cephas
 *
 * @see this class works like an interface between iris objects in Java and Weka instance needed
 * by classification method. it is a bit more than a parser but it function is restricted to build and
 * allow the classification with Weka ".model" file loaded.
 *
 */

public class IrisUtils {

	private Instances dataset;
	private ArrayList<Attribute> attributes;

	public IrisUtils() {
			createInstances();
		}

	private void createInstances() {
		this.dataset = new Instances("dataset", createAttributesToIris(), 0);
		this.dataset.setClassIndex(this.dataset.numAttributes() - 1);
		;
	}

	private ArrayList<Attribute> createAttributesToIris() {

		this.attributes = new ArrayList<Attribute>();

		attributes.add(new Attribute("sepalLength"));
		attributes.add(new Attribute("sepalWidth"));
		attributes.add(new Attribute("petalLength"));
		attributes.add(new Attribute("petalWidth"));

		// Create Class Attribute
		attributes.add(new Attribute("irisClass", createValuesForClass()));

		return attributes;
	}

	private List<String> createValuesForClass() {

		List<String> classValues = new ArrayList<String>();
		
		classValues.add(new String("Iris-setosa"));
		classValues.add(new String("Iris-versicolor"));
		classValues.add(new String("Iris-virginica"));

		return classValues;
	}

	public Instance irisToWekaInstance(Iris iris) {

		double[] values = new double[dataset.numAttributes()];

		values[0] = iris.getSepallength();
		values[1] = iris.getSepalwidth();
		values[2] = iris.getPetallength();
		values[3] = iris.getPetalwidth();

		values[4] = dataset.attribute(4).indexOfValue(iris.getIrisClass());

		Instance instance = new DenseInstance(1, values);

		return instance;
	}

	public Instances getDataset() {
		return dataset;
	}

	public void setDataset(Instances dataset) {
		this.dataset = dataset;
	}

}
