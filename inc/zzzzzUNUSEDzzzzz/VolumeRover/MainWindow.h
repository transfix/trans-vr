/*
	The QT GUI application starts with the execution of this main window widget.
	All the different functionalities present in the UI should be found in the
	documentation on how to use the user interface.
*/
#include <Qt3Support>

//Added by qt3to4:
#include <QCustomEvent>
#include <QCloseEvent>
#include <Q3BoxLayout>
#include <QLabel>
#include <QKeyEvent>
#include <QMainWindow>

class QTabWidget;
class VolumeInterface;
class VolumeViewerPage;

#if QT_VERSION < 0x040000
class QWidgetStack;
class QListView;
class QListViewItem;
#else
class QStackedWidget;
class QTreeWidget;
class QTreeWidgetItem;
class QWidget;
#endif

namespace Ui
{
  class MainWindowBase;
}


class MainWindow : public QMainWindow
{
		Q_OBJECT

	public:
		MainWindow(QWidget* parent = 0, const char* name = 0, Qt::WFlags f = Qt::WType_TopLevel);
		virtual ~MainWindow();


	public slots:

		virtual bool notYetImplementedSlot();

		virtual bool fileOpenSlot();
		virtual bool fileSaveSlot();
		virtual bool fileOptionsSlot();
		virtual bool fileExitSlot();

		/*

		virtual bool addNewDataSet(QStringList fileNames, bool separate = false);
		virtual bool deleteAllData();
		virtual bool deleteData(int dataSetIndex);
		virtual bool deletePrevData();
		virtual bool executeCommand(int argc, QStringList argv);

		virtual bool fileOpenMultiSlot();
		virtual bool saveImage(const char* fileName, const char* fileFormat);
		virtual bool setGridVisible(bool visibility);
		virtual bool setViewingParameters(double* translationParams, double* rotationParams, double windowSize);
		virtual bool setViewingParameters(int rendererId, double* translationParams, double* rotationParams, double windowSize);
		virtual bool setVisible(bool render, int dataSetIndex);
		virtual bool setVisiblePrev(bool render);
		virtual QString GB_write_area_file(QString in_filename);
		virtual QString GB_write_surface_file(QString in_filename);
		virtual void aboutHelpSlot();
		virtual void acknowledgementSlot();
		virtual void backgroundColorSlot();
		virtual void computePocketTunnelSlot();
		virtual void constructCurvaturesUtilitySlot();
		virtual void constructDepthColoredVolumeSlot();
		virtual void constructHLSPocketsSlot();
		virtual void constructNurbsSurfaceSlot();
		virtual void constructPocketsSlot();
		virtual void constructVolumeUtilitiesSlot();
		virtual void dataBoundingBoxSlot(bool);
		virtual void dataSetSelectedRightMouse(Q3ListBoxItem* qListBoxItem, const QPoint& qPoint);
		virtual void dataTransferProgressop(int bytesDone, int bytesTotal);
		virtual void deleteAndUpdateUI(int selectedIndex);
		virtual void dockingSlot();
		virtual void downloadPDBSlot();
		virtual void editLight(int light);
		virtual void elucidateSecondaryStructuresSlot();
		virtual void finishedop(bool b);
		virtual void forceMeshRenderingSlot(bool status);
		virtual void fullscreenSlot();
		virtual void GB_energy_Slot();
		virtual void GB_forceField_Slot();
		virtual void globalBoundingBoxSlot(bool);
		virtual void indexHelpSlot();
		virtual void matchSlot();
		virtual void newDataSetSelectedSlot(int);
		virtual void orthographicViewSlot();
		virtual void PB_energy_Slot();
		virtual void perspectiveViewSlot();
		virtual void playbackAnimationSlot();
		virtual void playbackMovieSlot();
		virtual void readPrevioslyOpenedFile(int previousFileIndexInMenu);
		virtual void recordAnimationSlot();
		virtual void redraw();
		virtual void renderTypeSlot(bool);
		virtual void saveImageSlot();
		virtual void scriptSlot();
		virtual void selectObjectsSlot(bool);
		virtual void setViewingParametersSlot();
		virtual void showGridOptionsSlot(bool);
		virtual void showMousePositionSlot(bool);
		virtual void showViewInformationSlot(bool);
		virtual void sliceParameterizationSlot();
		virtual void splitViewSlot();
		virtual void startRecordSlot();
		virtual void stereoSlot(bool);
		virtual void stopRecordSlot();
		virtual void surfaceDialogSlot();
		virtual void syncViewSlot();
		virtual void transformObjectSlot();
		virtual void updateUIWithNewData();
		virtual void userInteractionSlot();
		virtual void surfaceAreaAndVolumeDialogSlot();
		virtual void contentsSlot();
		virtual void resultSelectedInDockingSlot(int);

		*/

	protected:

		QTabWidget *_mainTabs;

		#if QT_VERSION < 0x040000
		  QWidgetStack *_volumeStack;
		  QListView *_volumeStackList;
		#else
		  QStackedWidget *_volumeStack;
		  QTreeWidget *_volumeStackList;
		#endif

		VolumeViewerPage *_volumeViewerPage;

		std::map<void*,VolumeInterface*> _itemToInterface;
		std::map<VolumeInterface*,void*> _interfaceToItem;

		void initializeMainWindowMenubar();
		QMenu *menuFile;

		// initialize Signal-Slot connections
		void initializeFileMenu();

		/*
		bool getSelected(BallAndStickData* & ballAndStickData, PDBParser::GroupOfAtoms* & molecule);
		bool getSelected(SurfaceData* & surfaceData, Geometry* & surface);
		//bool getSelected(BallAndStickData* ballAndStickData, Volume* volume);

		void setPreviousSavedEnvironment();
		void populateStatusBar();
		

		void initializeVariables();
		void initializeMainWindowStructures();
		

		void initializeViewMenu();
		void initializeAnimationMenu();
		void initializeOptionsMenu();
		void initializeUtilitiesMenu();
		void initializeHelpMenu();

		// UI functions
		void updateListBox();
		void createLightsMenu();
		bool enableFirstLight();

		void setChildrenVisible(bool mode);
		void setRendererFullScreen(bool mode);
		void keyPressEvent(QKeyEvent* event);
		void keyReleaseEvent(QKeyEvent* event);
		void closeEvent(QCloseEvent* event);
		void debugPrint();
		void buildMovie();

		enum POPUPSELECTION {SAVE_DATASET,DELETE_DATASET};
		POPUPSELECTION showPopup(QPoint point);
		void customEvent(QEvent* event);

		*/

		// mainwindow ui
		QWidget* _centralWidget;
		Ui::MainWindowBase *_ui;

		
		//*****************************
		//	Menu Actions
		//*****************************

		QAction *m_Action_FileOpen;
		QAction *m_Action_FileSave;
		QAction *m_Action_FileOptions;
		QAction *m_Action_FileExit;


		/*
		QAction *m_Action_FileOpenMulti;
		QAction *m_Action_DownloadPDB;
		QAction *m_Action_SaveImage;
    		QAction *m_Action_Exit;
    		QAction *m_Action_RecentFileListDummy;
    		QAction *m_Action_OrthographicView;
    		QAction *m_Action_PerspectiveView;
  		QAction *m_Action_SetViewParameters;
    		QAction *m_Action_SplitView;
    		QAction *m_Action_SyncView;
    		QAction *m_Action_FullScreen;
    		QAction *m_Action_Stereo;
    		QAction *m_Action_ForceMeshRender;
    		QAction *m_Action_TransformObject;
    		QAction *m_Action_StartRecording;
    		QAction *m_Action_StopRecording;
    		QAction *m_Action_PlaybackAnimation;
    		QAction *m_Action_RecordAnimation;
    		QAction *m_Action_PlaybackMovie;
    		QAction *m_Action_RayTrace;
    		QAction *m_Action_BackgroundColor;
    		QAction *m_Action_DisplayGrid;
   		QAction *m_Action_MouseKeyboardFunc;
    		QAction *m_Action_SelectObjects;
    		QAction *m_Action_GlobalBoundingBox;
    		QAction *m_Action_DataBoundingBox;
   		QAction *m_Action_ShowViewInformation;
	 	QAction *m_Action_ShowMousePosition;
    		QAction *m_Action_Script;
		QAction *m_Action_ConstructSurface;
    		QAction *m_Action_ConstructNURBSSurface;
    		QAction *m_Action_SurfaceAreaAndVolume;
   		QAction *m_Action_ConstructVolume;
    		QAction *m_Action_ConstructDepthColoredVolume;
    		QAction *m_Action_ConstructPockets;
    		QAction *m_Action_ConstructPocketTunnelStableManifold;
    		QAction *m_Action_ConstructHLSPockets;
    		QAction *m_Action_GetCurvature;
    		QAction *m_Action_FormMatch;
    		QAction *m_Action_ComputeEnergyGB;
    		QAction *m_Action_ComputeForceFieldGB;
    		QAction *m_Action_ComputeEnergyPotentialPB;
    		QAction *m_Action_Get2DSlice;
    		QAction *m_Action_ElucidateSecondaryStructure;
    		QAction *m_Action_F2Dock;
    		QAction *m_Action_Contents;
    		QAction *m_Action_Index;
    		QAction *m_Action_Acknowledgements;

		QMenu *menuFile;
    		QMenu *m_Action_RecentFileListMenu;
    		QMenu *menuView;
    		QMenu *menuProjection;
    		QMenu *menuAnimation;
    		QMenu *menuOptions;
    		QMenu *m_Menu_LightingPopup;
    		QMenu *menuUtilities;
    		QMenu *menuHelp;
		//*****************************
		
		Q3Http* http;
		QFile* http_file;
//		Q3BoxLayout* m_QBoxLayout;
		QGridLayout* m_QBoxLayout;
		DataManager* m_DataManager;
		QWidget* m_PropertyWidget;
		RendererSet* m_RendererSet;
		Documentation* m_Documentation;
		Settings* m_Settings;
		ScriptsDialog* m_ScriptsDialog;
		Server* m_Server;
		QLabel* m_ViewInformationLabel;
		QLabel* m_MouseInformationLabel;
		QLabel* m_SelectionInformationLabel;
		bool m_FullScreen;
		bool m_SyncView;
		MouseHandler* m_MouseHandler;
		LightSet* m_LightSet;
		QSProject* m_QSProject;

		vector<double *> transformations;
		vector<int> result_indices;
		int numSelectedResults;
		string receptorName;
		string ligandName;
		PDBParser::GroupOfAtoms* ligandMolecule;
		PDBParser::GroupOfAtoms* receptorMolecule;
		Geometry* ligandSurface;
		Geometry* receptorSurface;
		SurfaceData *ligandSurfaceData;
		SurfaceData *receptorSurfaceData;
		BallAndStickData *ligandUofBData;
		BallAndStickData *receptorUofBData;
		bool receptorRendered;
		bool surfaceGenerated;
		bool PBGenerated;
		bool GBGenerated;
		bool CoulGenerated;
		QStringList dockingReceptorList;
		QStringList dockingOutputList;


		int m_F2DockReceptorIndex;
		int m_F2DockLigandIndex;
		int m_F2DockRendererId;
		int m_F2DockReceptorSurfaceIndex;
		int m_F2DockLigandSurfaceIndex;
		QString m_ReceptorXYZ;
		QString m_LigandXYZ;
		QString m_ReceptorRAW;
		QString m_LigandRAW;

		*/

};

