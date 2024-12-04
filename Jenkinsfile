pipeline {
	 agent any
	 stages {
	 	stage('Hello'){
			steps {
			      echo 'Hello World'
			      }
		}
		stage('Testing'){
			steps {
			      bat(script: 'py -m pytest --junitxml results.xml tests.py')
			      }
		}
	}
	post {
	     always {
	     	    junit(testResults: 'results.xml', allowEmptyResults : true)
		    }
		    
    	     }
}


