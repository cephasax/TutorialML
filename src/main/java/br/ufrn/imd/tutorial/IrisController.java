package br.ufrn.imd.tutorial;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import weka.core.Instance;

@RestController
@RequestMapping("/iris")
public class IrisController {

	@Autowired
	private MachineLearningModel mlm;

	//New Iris flower classification
	@RequestMapping(method = RequestMethod.POST,headers = "Content-Type=application/json")
	public Iris classifyIrisFlowe(@RequestBody Iris i) {
		Iris iris = i;

		String classValue = new String(classifyIrisFlower(iris));
		iris.setIrisClass(classValue);
		
		return iris;
	}
	
	private String classifyIrisFlower(Iris iris) {
		IrisUtils iu = new IrisUtils();
		
		Instance instance = iu.irisToWekaInstance(iris);
		
		iu.getDataset().add(instance);
		
		double value = -1;
		try {
			value = mlm.getCls().classifyInstance(iu.getDataset().get(0));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//get the irisClass value
		String prediction = null; 
		prediction = new String(iu.getDataset().classAttribute().value((int)value));
		
		return prediction;

	}
	
}
