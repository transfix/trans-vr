#include <qapplication.h>
#include <AnimationMaker/AnimationWidget.h>
#include <VolumeWidget/ZoomInteractor.h>
#include <VolumeWidget/TrackballRotateInteractor.h>
#include <VolumeWidget/PanInteractor.h>
#include <AnimationMaker/MouseSliderHandler.h>
#include <VolumeWidget/WireCubeRenderable.h>
#include <GeometryFileTypes/GeometryLoader.h>
#include <AnimationMaker/RawIVTestRenderable.h>


int main( int argc, char** argv ) {
	QApplication::setColorSpec( QApplication::ManyColor );
	QApplication app( argc, argv );

	AnimationWidget animationWidget;
	app.setMainWidget(&animationWidget);

	/*
	GeometryLoader loader;


	//Geometry* geometry = loader.loadFile("1MaCHE_pot.rawnc");
	//Geometry* geometry = loader.loadFile("mache_0_6_hydro.rawnc");
	//Geometry* geometry = loader.loadFile("mache_0_6_hydro1.rawnc");
	Geometry* geometry = loader.loadFile("myoutput0.rawnc");


	//geometry->SetAmbientColor(0.1f, 0.1f, 0.1f);
	WireCubeRenderable wireCubeRenderable;
	RawIVTestRenderable rawVTestRenderable;
	animationWidget.initForContext(&rawVTestRenderable);
	//if (rawVTestRenderable.setFileName("ribosome-1JJ2.rawv")) {
	//	animationWidget.setMainRenderable(&rawVTestRenderable);
	//	rawVTestRenderable.enableVolumeRendering();
	//}
	//else 
	if (geometry) {
		GeometryRenderable* geometryRenderable = new GeometryRenderable(geometry);
		animationWidget.setMainRenderable(geometryRenderable);
	}
	else {
		animationWidget.setMainRenderable(&wireCubeRenderable);
	}
	*/

	FILE* fp;
	if (fp=fopen("animation.txt", "r")) {
		animationWidget.playBackFrom(fp);

	}
	else {
		fp = fopen("animation.txt", "w");
		//animationWidget.setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, new PanInteractor);
		animationWidget.setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, new MouseSliderHandler(&(animationWidget.m_NearPlane)));
		animationWidget.setMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler, new ZoomInteractor);
		animationWidget.setMouseHandler(SimpleOpenGLWidget::WheelHandler, new ZoomInteractor);
		animationWidget.setMouseHandler(SimpleOpenGLWidget::RightButtonHandler, new TrackballRotateInteractor);
		animationWidget.recordTo(fp);
	}
	

	
	animationWidget.show();
	//animationWidget.showFullScreen();

	return app.exec();

}
