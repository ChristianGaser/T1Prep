function cat24_bc(job2)
  % cat24 bias correction
  % - I tested our Bestoff sample and some others
  % - It should make no big difference in the resulting quality 
  %   which space (native/resampled) or if denoising was used 
  % - Runtime grows a bit stronger for higher resolutions, e.g., 
  %     IXI ~ 10s, HR075 ~ 30s, yv98_mprage_05mm ~ 86s
  %
  % = ADNI and OASIS background are not nice but within the brain ok !
  % - subcortical structures could be better
  
  job.verb             = 1; % display progress
  job.dt               = 2; % SPM image datattype (16  - float, 512 - int16, 2 - uint8)
  job.normInt          = 1; % intensity normalization 
  job.simpleApprox     = 0; % use simple approximation rather than cat_vol_approx
  job.write.org        = 1; % write intensity normed image (for debugging)
  job.write.GAS        = 1; % do and write global adapted segmentation (GAS) map 
                            % 1 - basic, 2 - fine
  job.write.LAS        = 1; % do and write local adapted segmentation (LAS) map
                            % this is a bit inoptimal for the histogram but helps visually
                            % however, in some protocols it could be also helpful (eg. MP2Rage) 
  job.write.biasfield  = 1; % write bias field map (for debugging) 
  
  if exist('job2','var')
    job = cat_io_checkinopt(job2,job); 
    P = job.files; 
  else
    % #### just my manual test case - remove later ####
    P = {
      '01_Bestoff/4397_T1_thinWM'                                   % 1 - could be better
      '01_Bestoff/Aarhus_TorbenLund_R1_0001'                        % 2  
      '01_Bestoff/ADHD200NYC_T1_defacedMoveArt'                     % 3
      '01_Bestoff/ADNI_AD_100_0743_T1_sd000000-rs00_S17224_I62363'  % 4
      '01_Bestoff/ADNI_AD_114_0228_T1_sd000000-rs00_S11697_I49735'  % 5 - BG sphere issue !!!
      '01_Bestoff/BUSS02_T1_infant'                                 % 6 - subcortical?
      '01_Bestoff/Cathy6_T1_hydrocephlus'                           % 7 - UPSI ... !!!
      '01_Bestoff/HR075_MPRAGE'                                     % 8
      '01_Bestoff/Hendrik_FLAIR'                                    % 9
      '01_Bestoff/HR075_T2'                                         % 10 - subcortical in Yml worse
      '01_Bestoff/Magdeburg_TilmannKlein_S26_mp2rage'               % 11
      '01_Bestoff/OASIS1_AD_001_0031_T1_sd000000-rs00'              % 12
      '01_Bestoff/SRS_R0500my_Tohoku_VBM8bo'                        % 13
      ...'01_Bestoff/Leipzig_Bazin_MP2RAGE_2017_194140_T1w'            % 14 missing label
      '01_Bestoff/IXI_HC_HH_175_T1_SD000000-RS00'                   % 15
      '01_Bestoff/IXI_HC_IO_543_T1_SD000000-RS00'                   % 16
      ...
      '04_MR_Protocols/yv98_mprage_05mm'                            % 17
      '04_MR_Protocols/Magdeburg7T_skc73'                           % 18
      '04_MR_Protocols/Gorgolewski2017_T1_7T'                       % 19
      '04_MR_Protocols/IXI175-HH-1570-MRA'                          % 20
      '04_MR_Protocols/London_GabrielZiegler_20160303_NSPNID_10736_00_MT' % 21
      '04_MR_Protocols/London_GabrielZiegler_20160303_NSPNID_10736_00_R1' % 22
      '04_MR_Protocols/rsub-bhl522_T1w_rs'
      '04_MR_Protocols/Gorgolewski2017_T2_7T'
      '04_MR_Protocols/FlaviaArduini_008_t2_tse_sag_volumetrica_iso'
      '04_MR_Protocols/Colin27_T2'
      '04_MR_Protocols/Colin27_T1'
      '04_MR_Protocols/Colin27_PD'
      '04_MR_Protocols/Cathy0_T1_MP2U'
      '04_MR_Protocols/Cathy0_T1_MP2'
      '04_MR_Protocols/Cathy0_T1_MP1'
      '04_MR_Protocols/Cathy0_PD_MP2'
      '04_MR_Protocols/Hanke2015_T2_7T'
      '04_MR_Protocols/Hanke2015_T1_7T'
      '04_MR_Protocols/yv98_05mm_corrected_division_n3' % CSF-LAS-bc
      };
  end
  six   = 1:numel(P);  % subjects
  
  
  %%
  for si = six(1:end)     
    % si=1;
  
    fprintf('%s\n',P{si}); stime2 = clock; 
    stime = cat_io_cmd('  Prepare data','g5'); 
    
    % input filenames
    if exist('job2','var')
      Po  = spm_file(P{si},'suffix','_desc-sanlm');
      Pl  = spm_file({si},'suffix','_desc-high_label');
      Pa  = spm_file(P{si},'suffix','_desc-high_atlas');
    else
      Po  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_desc-sanlm.nii', P{si});
      Pl  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_res-high_label.nii', P{si}); 
      Pa  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_res-high_atlas.nii', P{si});
    end
    % output filenames 
    Po2 = spm_file(Po,'suffix','_org');
    Pm  = spm_file(Po,'suffix','_bias');
    Pml = spm_file(Po,'suffix','_las');
    % the bias field maps are for debugging
    Pms = spm_file(Po,'suffix','_lasseg');
    Pbf = spm_file(Po,'suffix','_biasfield');
    Plf = spm_file(Po,'suffix','_biasfieldlas');
  
    % TEMPORARY if the denoised file is missing than use the original one
    if ~exist(Po,'file')
      Po  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s.nii', P{si});
    end
  
    % load header and images depending on the image space
    native = 1; % 0 - normalized label maps space, 1 - native space of the Yo 
    Vo  = spm_vol(Po); 
    Vl  = spm_vol(Pl);
    Va  = spm_vol(Pa);
    if native
      Yo  = single(spm_read_vols(Vo)); 
      Vp0 = Vo; Vp0.fname = spm_file(Vp0.fname,'suffix','_seg-native');
      [Vp0,Yp0] = cat_vol_imcalc([Vo,Vl],Vp0,'i2');
      Va0 = Vo; Va0.fname = spm_file(Va0.fname,'suffix','_atlas-native');
      [Va,Ya]   = cat_vol_imcalc([Vo,Va],Va0,'i2',struct('interp',0));
    else
      Yp0 = spm_read_vols(Vl); 
      Ya  = spm_read_vols(Va); 
      Vp0 = Vp0; Vp0.fname = spm_file(Vo.fname,'suffix','_org-mri');
      [Vo,Yo] = cat_vol_imcalc([Vl,Vo],Vp0,'i2');
    end
  
  
    % ### Subject / image information ### 
    % These are some simple basic information about the image and subject. 
    % get single tissue class from label map (CSF=1, GM=2, WM=3)
    Yp0toC = @(Yp0,c) max(0,1-min(1,abs(Yp0-c)));
    vx_vol = sqrt(sum(Vo.mat(1:3,1:3).^2));   
  
  
    % Gradient edge map with average tissue intensity (eg. double it to threshold)
    % Edge maps can be easily normalized Yg./T3th or Yg./Yo and are then bias
    % free because of their local nature!), as the normalization can change
    % it is not inlduced here yet. The mimimum is used to avoid aging bias, 
    % with more 'noisy' WM by perivescular spaces in elderly but at once huge 
    % ventricles that are on the other side not available in young subjects. 
    % Nevertheless, we will try to avoid outliers with "Yg(:)./Yo(:)<.5".
    T3th  = [cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),1)>.9 )) ...
             cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),2)>.9 )) ...
             cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),3)>.9 ))];
    bias  = cat_stat_nanstd( Yo( Yp0toC(Yp0(:),3)>.9 )) ./ min(diff(T3th));
    Yg    = cat_vol_grad(Yo) .* cat_vol_morph(Yo~=0,'e',1);
    Yg    = Yg ./ max( Yo , max( max(T3th)/5 , min(T3th)/2 )); %* (0.25/bias)
    gth   = max( 0.05 , min( ...
             [ cat_stat_nanmedian( Yg( Yp0toC(Yp0(:),1)>.95 & Yg(:)<.9)) , ...
               cat_stat_nanmedian( Yg( Yp0toC(Yp0(:),3)>.95 & Yg(:)<.9)) ] ));
   
  
    % relative volumes (eg. to handle enlarged ventricles/lession or failed processing) and 
    % TIV (eg. to generally handle children or other species or failed processing) 
    T3vol = [cat_stat_nansum( Yp0toC(Yp0(:),1) ) ...
             cat_stat_nansum( Yp0toC(Yp0(:),2) ) ...
             cat_stat_nansum( Yp0toC(Yp0(:),3) )];
    T3tiv = sum(T3vol) .* prod(vx_vol/10); 
    T3vol = T3vol ./ sum(T3vol);
    
  
    hydrocephalus = T3vol(1) > .25; % we need to take of large ventricle 
  
  
    % Estimate tissue thresholds of major structures to avoid bias by
    % missclassfied objects and larger PVE areas.  
    Ywm   = cat_vol_morph( Yp0toC(Yp0,3)>.9 & Yg<gth*2 , 'l' , [10 0.1]) > 0; 
    Ygm   = Yp0toC(Yp0,2)>.9 & ~cat_vol_morph( Yp0toC(Yp0,2)>.5 ,'o',3); 
    Ycm   = cat_vol_morph( Yp0toC(Yp0,1)>.9 & Yg<gth*2 , 'lo' , 1);
    if sum(Ycm)<1000, Ycm = Yp0toC(Yp0,1)>.9; end
    T3th  = [cat_stat_nanmedian( Yo( Ycm(:) )) ...
             cat_stat_nanmedian( Yo( Ygm(:) )) ...
             cat_stat_nanmedian( Yo( Ywm(:) ))];
    if 1
      [hn,hi] = hist( Yo(Yo(:)~=0)  , 10000);
      hni   = find( cumsum(hn) / max(cumsum(hn))>.0001 ,1);
      Tmin  = min( [ 0 , hi(hni) ] );
    end
    
    % estimate image contrast (T1w/T2w/PDw) - finally not used here
    if     T3th(1) < T3th(2) && T3th(2) < T3th(3), modality = 1;   % T1w
    elseif T3th(1) > T3th(2) && T3th(2) > T3th(3), modality = 2;   % T2w
    elseif T3th(1) < T3th(3) && T3th(2) < T3th(3), modality = 0;   % PDw - WM is maximum
    else                                         , modality = 3;   % PDw - WM is minimum
    end
  
  
  
  
    %% Fast bias correction and background estimation
    % Some inital very low-frequent tissue unspecific bias-correction that we 
    % need for masking and especially for the general correction in the head
    % and backround.
    stime = cat_io_cmd('  Initial bias correction','g5','',job.verb,stime); 
    
  
    % First, we have to catch some special cases of our given segmentation.
    % We have to remove miss-classified tissue in extrem cases (hydrocephalus).
    % For WMHs we have to avoid until some basic bias-correction.
    if hydrocephalus 
      % this operation restore large ventricle but further test in other special cases are required
      Ymskv = smooth3( Yp0>2 & Yo/mean(T3th(1:2))<Yp0/3 ) > 0.5; 
      Ymskv = cat_vol_morph( cat_vol_morph(Yp0toC(Yp0,1)>.5,'lo',2) | Ymskv , 'ldc' , 4); 
      Yp0c  = min(Yp0,3 - 2*smooth3(Ymskv)); 
      Yp0c  = max(Yp0c, cat_vol_morph( Yp0>=1 ,'lc')); 
    else
      Yp0c  = Yp0; 
    end
    % Remove objects within the CSF. ####
    Yp0c(Yp0>0  & Yp0<1.0 & Yg>gth  ) = 0; % meninges (CSF-bone/background boundary) 
    Yp0c(Yp0>=1 & Yp0<1.5 & Yg>gth*2) = 0; % meninges, blood-vessels, ...
    
  % ### better to have the ventricle separatelly, because CSF values sometimes differ here. ####
  % ### what to do with edgy GM, namely in the cerebellum? > no, replace later #####
  
    % Sedond, we need a intensity normalized version of the Yp0 adopted for the
    % the image tissue intensies. The outer CSF may include meninges or bone
    % and is better been removed! 
    Yp0t = zeros(size(Yp0c)); for ci = 1:3, Yp0t = Yp0t + T3th(ci) .* Yp0toC(Yp0c,ci); end
    Yp0t = Yp0t .* cat_vol_morph(Yp0c>1,'lo',2); 
  
  
  
  
    %% Head segmentation that could be replace by something else in future:
    %  We need to guess about head tissues in general to estimate the bias field. 
    %  Therefore, we have to segment the background/object. It would be cool to
    %  separate between the head tissues but how?
    %  However, first we have to get the object:
    
    % We can do this on lower resolution, what supports also noise/artifact reduction. 
    [Ygr,Yor,Yp0r,Yp0tr,rs] = cat_vol_resize( {Yg,Yo,Yp0c,Yp0t} , 'reduceV',vx_vol,2,16,'meanm');
  
    % For basic correction we have to separate object (include information for
    % correction) and background (limited use and critical due to artifacts).
    Yhdr  = smooth3( Ygr ) > gth | Yp0r>0; 
    Yb0   = false(size(Yor)); Yb0(2:end-1,2:end-1,2:end-1) = true; % required for opening
    Yhdr  = cat_vol_morph( Yhdr & Yb0, 'ldo', 8 ); % avoid background objects
    Yhdr  = cat_vol_morph( Yhdr , 'ldc', 8, rs.vx_volr); % fill head (huge and slow operation)
  
    % detect and hadle MP2Rage backgrounds 
    Ymp2r = Yp0r==0 & ~Yhdr & Yor>min(T3th) & Yor<max(T3th) & Ygr < 0.5 & Ygr~=0;
    % separate low and high intensity tissues ##### too T1w specific?
    Yp0d  = cat_vol_smooth3X(Yp0r>0,4);
    Ytrl  = cat_vol_morph( smooth3(Yor>median(T3th)/10 & Ygr<gth*4 & Ygr~=0 & Yhdr & Yp0d<0.05)>.4,'o') & ~cat_vol_morph(Ymp2r,'dd',4);
    % skull-stripped/defaced
    Ybg0r = cat_vol_morph(Yor==0 | Yor==min(Yor(:)),'o',2); 
    % create image with intensity definition of known issues (head label map)
    Ytr   = Ymp2r * cat_stat_nanmedian([0;Yor(Ymp2r(:))]) ...  MP2RageBG
            + Ytrl * max( [ min(T3th) , median(T3th).*(modality==1) , ...
              cat_stat_nanmedian(Yor(Ytrl(:))).* (modality~=1)])  ...  head tissue 
            + Yp0tr ... intensity scaled brain
            + Ybg0r * T3th(3)*0.01; % empty background
    Yi    = cat_vol_localstat( Yor ./ max(eps,Ytr) .* (Ytr>0) , Ytr>0, 1, 3); 
    if job.simpleApprox == 1
      Ywr = simple_approx( Yi );
    else
      Ywr = cat_vol_approx( Yi );
    end
    Ywrs  = cat_vol_smooth3X(Ywr,min(32,max(8,4 * 1/bias))); 
    Ywr   = Ywrs ./ median(Ywrs(Yp0r(:)>0)) .* median(Ywr(Yp0r(:)>0));
    Ymr   = Yor ./ Ywr; 
    Ymr   = Ymr ./ cat_stat_nanmedian(Ymr(Yp0toC(Yp0r,3)>.9)); % final scaling  
    Ymr(Ybg0r) = 0; % keeping skull-stripped/defaced backgound
    
    
  
  
    %% Second correction level where we can use the soft bias corrected image. 
    % postcorrection of object/background
    Yhdr  = Yhdr & Yb0 & smooth3(Ymr)>.5; 
    Yhdr  = cat_vol_morph( Yhdr, 'ldo', 1 );
  
    % Now we can handle WMHs and subcortical structures that we do not want 
    % to correct in the bias correction at all. 
    Ywmh  = Yp0r>2 & Yp0r<=3 & smooth3(Yp0r - Ymr*3)>.2;
    Ybg   = Yp0r>=2 & Yp0r<2.1 & smooth3(Ymr*3 - Yp0r)>.2; 
    Ybg(smooth3(Ybg)<0.3) = 0; Ybg = cat_vol_morph(Ybg,'d');
    Yp0tr(Ywmh | Ybg) = Ymr(Ywmh | Ybg) * T3th(3);
    % Ymp2r = ~Yhdr & (Ymr>0.3 & Ymr<1.1 & Ygr < 0.5); % add MP2R background
    Ytrl  = (Ygr<gth*4) & Ygr~=0 & (Ymr>.5) & (Ymr<.9) .* Yhdr .* (Yp0r==0); Ytrl(smooth3(Ytrl)<.2) = 0;
    Ytrh  = (Ymr>1.2)  & Ygr~=0 & Yhdr & (Yp0r==0); Ytrh(smooth3(Ytrh)<.2) = 0;
  
    Ytr   = Ymp2r .* (~Ytrl & ~Ytrh & ~Ybg0r) .* cat_stat_nanmedian([0;Ymr(Ymp2r(:))]) ...
            + Ytrl .* cat_stat_nanmedian([0;Ymr(Ytrl(:))]) ...
            + Ytrh .* cat_stat_nanmedian([0;Ymr(Ytrh(:))]) ...
            + Yp0tr/T3th(3) ...
            + Ybg0r*0.0001 ...
            + 0; 
    Yi    = cat_vol_localstat( Yor ./ max(eps,Ytr) .* (Ytr>0) , Ytr>0, 1, 3); 
    if job.simpleApprox == 1
      Ywr = simple_approx( Yi );
    else
      Ywr = cat_vol_approx( Yi );
    end
    Ywrs  = cat_vol_smooth3X(Ywr,min(8,max(4,2 * 1/bias)) / mean(rs.vx_volr) );  
    Ywr   = Ywrs ./ median(Ywrs(Yp0r(:)>0)) .* median(Ywr(Yp0r(:)>0));
    Ymr   = Yor ./ Ywr;
  
    % final scaling (similar to Yo!)
    Ymr   = Ymr ./ cat_stat_nanmedian(Ymr(Yp0toC(Yp0r,3)>.9));
    Ymr(Ybg0r) = 0; % keeping skull-stripped/defaced backgound
  
    % final scaling 
    %Yhd0  = cat_vol_resize( smooth3(Yhdr) , 'dereduceV', rs) > .5; 
    Yw0   = cat_vol_resize( smooth3(Ywr)  , 'dereduceV', rs);
    Yw0   = cat_vol_smooth3X(Yw0,4); Yw0 = Yw0 ./ median(Yw0(Yp0(:)>0)) .* median(Yw0(Yp0(:)>0));
   % clear Ygr Yor Yp0r;
    Ym0   = Yo ./ Yw0; 
    Ym0   = Ym0 ./ cat_stat_nanmedian(Ym0(Yp0toC(Yp0,3)>.9));
    
  
    % udpate tissue thresholds
    T3thc = [cat_stat_nanmedian( Ym0( Ycm(:) )) ...
             cat_stat_nanmedian( Ym0( Ygm(:) )) ...
             cat_stat_nanmedian( Ym0( Ywm(:) ))];
   
    % Background and Skull-Stripping cases
    %{
      hdth  = cat_stat_nanmedian(Yo(Yhd0)); 
      hdvr  = sum(Yp0(:)>0) ./ sum( Yhd0(:) & ~Yp0(:)>0); 
      highbackground = hdth / T3th(3) > .5 & hdvr<0.3;
      skullstripped  = T3tiv < 2500 & hdth==0 & hdvr>.9;
    %}
  
  
  
    %% (1) Optimization of the label map for bias correction:
    %  ----------------------------------------------------------------------
    % Yp0c is a corrected label map, where we align/set no-cortical GM voxels, 
    % i.e., very thick regions, to a partial volume class/intensity, to avoid
    % overcorrections of the subcortical structures and within the cerebellum.
    % We use here a value of 2.5 as we do not have better information yet.
    % Alternatively, such voxel could be ignored and set to zero. 
    stime = cat_io_cmd('  Optimize labels for bias correction','g5','',job.verb,stime); 
    
    % Fast approximation of a relative skull-core map 
    % This map can be used to roughly separate cortial from subcortical GM  
    voli = @(v) (v ./ (pi * 4./3)).^(1/3); 
    rad  = voli( T3tiv) ./ cat_stat_nanmean(vx_vol);
    Ysc  = 1-cat_vol_smooth3X(Yp0<1 | Yo==0,min(24,max(16,rad*2)));   % fast 'distance' map
    
    % get deep GM by removing thinner cortical structures (thickness below 3 mm) 
    Ydgm = Ysc>.9 & cat_vol_morph( Yp0toC(Yp0,2)>.5, 'do', 3, vx_vol); 
    
    Yp0c = Yp0; 
    % (a) handling of GM/WM (PVE) tissues (labeled as GM but with WM like values) 
    if 1
      % Smooth correction of no-cortical GM by using the maximum function.
      % We assume here 2.5 as average class (we cannot use Yo as it is not 
      % bias corrected yet).
      Ydgw = cat_vol_morph( Ydgm | Yp0>2.5,'c') & Yp0>=2;
      Ydgw = cat_vol_smooth3X( Ydgw ,4); 
      Yp0c = max(Yp0c, 2*(Yp0c>2 & Ydgw>0.1) + .5*Ydgw);
      clear Ydgw; 
    else
      % ignore them
      Yp0c(Ydgm) = 0; 
    end
    % (b) handling of WMHs (labeled as WM but with GM like values)
  % ###### not required yet ######
  
    % we also have to remove miss-classified tissue
    if hydrocephalus 
      Ymskv = smooth3( Yp0>2 & Yo/mean(T3th(1:2))<Yp0/3 ) > 0.5; 
      Yp0c  = min(Yp0c,3 - 2*smooth3(Ymskv)); 
    end
  
    % It is highly important to also consider the head for bias correction. 
    % As we have no head segmenation we use the general bias correction for 
    % the wider area around the brain.
    
    % this map is for other modalities
    [T3thord,Tord] = sort(T3thc);
    Yp0t = zeros(size(Yp0c)); for ci = 1:3, Yp0t = Yp0t + T3thord(ci) .* Yp0toC(Yp0c,ci); end
  
  
    %% (2) classical bias correction 
    %  ----------------------------------------------------------------------
  
    % Yi is a map with intensity values used for later bias field approximation
    % defined by Yw.  A multiplicative bias quite smooth bias correction model is used. 
    stime = cat_io_cmd('  Bias correction','g5','',job.verb,stime); 
  
    % define object/tissue (here we need a weighting specific label map Yp0t !)
    Yi = (Yg<.5 & Yg~=0 & Yp0>=1 & Yp0c > 0.5) .* single( Yo ./ (Yp0t) );
    Yi(Ym0 < min(T3thc)/max(T3thc)*.9 | Ym0 > 1.5 | (Yp0<1.2 & Yg>gth)) = 0; % remove crazy tissue values
    Yi( Yi./Yw0 < 0.8  |  Yi./Yw0 > 1.5 ) = 0; 
    Yi(smooth3(Yi>0)<.5) = 0;
  %  Yi( abs(Yp0t - min(1,Ym0)) > 0.1 ) = 0; % remove crazy segmentation areas
    Yi( Yo == min(Yo(:))) = T3th(3); % GZ-MT
    Yi( cat_vol_morph( Yo == max(Yo(:)),'d')) = 0; % GZ-MT
   
    if 0 %hydrocephalus 
    % in case of hydrocephalus parts of the ventricles are maybe miss-
    % classified as tissue. However, the CSF estimation should be fine. 
      Ymskh = cat_vol_morph( Ysc>.8 & Yp0toC(Yp0,1)>.5 & Yg < gth * 4, 'lo') ; 
      Yi(Ymskh) = Yo(Ymskh) / median(Yo(Ymskh(:))) * T3th(3); 
  
      % we also have to remove miss-classified tissue
      Yi( Ymskv ) =  Yo(Ymskv) / median(Yo(Ymskv(:))) * T3th(3); 
  % ##########
  % In case of misslabeling we have to use smoother maps.
  % ##########
    end
    %
    if 1
    % refined preparation (takes a while but helps to fix some corners)
    
      % remove noisy structures
      Yi(smooth3(Yi)<0.8) = 0; 
      
      % use median filter to remove outliers (this is important for the maximum filter)
  %    Yi = cat_vol_median3(Yi,Yi>0 & Yi<max(Yi(:)),Yi>0 & Yi<max(Yi(:)));
      
      % use maximum filter to avoid PVE values to non object voxels
      Yi = cat_vol_localstat(Yi,Yi>0 & Yi<max(Yi(:)),1,3,2); % must be 2
      Yi = cat_vol_localstat(Yi,Yi>0 & Yi<max(Yi(:)),1,1,1); % 
    end
    % bias field with some distance 
    Ywbg = (cat_vol_smooth3X(Yp0,4)<.01) .* Yw0 ./ ...
      cat_stat_nanmean(Yw0(Yi>0 & Yp0toC(Yp0,3)>.9)) * cat_stat_nanmean(Yi(Yi>0 & Yp0toC(Yp0,3)>.9));
    Yi = max(Yi,Ywbg); 
    %
     % approximation of un-defined voxels and creation of a smooth map
    if job.simpleApprox == 1
      Yw = simple_approx( Yi );
    else
      Yw = cat_vol_approx( Yi , 'nn' , vx_vol, 2); 
    end
    Yws = cat_vol_smooth3X(Yw,6); Yws = Yws ./ median(Yws(Yp0(:)>0)) .* median(Yw(Yp0(:)>0));
    % smoother close to large GM structures
    Ymx = min( 1, cat_vol_smooth3X( cat_vol_smooth3X(Yp0>0,8)>.1,8) );
    Yw  = Yw.*(1 - Ymx) + Ymx .* Yws;
    % the field has to be smooth to support further corrections !
    Ym  = Yo ./ Yw; 
    Ym  = (Ym - Tmin) ./ (cat_stat_nanmedian(Ym(Yp0toC(Yp0c,3)>.9 & Yp0toC(Yp0,3)>.9)) - Tmin);
    Ymx = cat_vol_smooth3X(Ydgm,16);
    
    
  
  
    %% (3) Local Adaptive Segmenation (LAS)
    %  ----------------------------------------------------------------------
    %  This function estimates the local tissue intensity based within a masked
    %  area (eg. the WM), given by a standard tissue segmentation, and approximates
    %  values outside the max and smoothes values within the mask. 
    %  - This part is already working quite well -
    if job.write.LAS 
      stime = cat_io_cmd('  Local Adaptive Segmenation (LAS)','g5','',job.verb,stime); 
      
      Ysrc   = single( Ym ); % would be possible to use also the orignal (i.e., Yo/T3th(3) )
      spmcls = [3 1 2];      % SPM class definition
      Ylab   = cell(1,6);    % SPM class tissue maps with local tissue intensity
      [~,T3ths] = sort(T3th,'descend'); cio = 0; 
      for ci = T3ths % 1:3 % CSF - GM - WM
        % extract tissue intensity (first by the major class, second avoid PVE regions)
        % small tissue threshold cause problems with PVE
        if spmcls(ci) == 1 % GM
          Yi = Yp0toC(Yp0,ci) > 0.8  & Yp0toC(Yp0c,ci) > 0.8  &  abs(Ym-Yp0/3) < 0.5; 
        elseif spmcls(ci) == 2 % WM
          Yi = Yp0toC(Yp0,ci) > 0.8  & Yp0toC(Yp0c,ci) > 0.8  &  abs(Ym-Yp0/3) < 0.15; % av
          Yi = cat_vol_morph(Yi,'de',1.5,vx_vol);
        else
          Yi = Yp0toC(Yp0,ci) > 0.8  & Yp0toC(Yp0c,ci) > 0.8  &  abs(Ym-Yp0/3) < 0.5; % av
          Yi = cat_vol_morph(Yi,'de',1.5,vx_vol);
        end
        Yi(smooth3(Yi)<0.5) = 0; 
        Yi = Yi .* Ysrc; 
      

        % approximation of unmasked values with smoother tissues
        if job.simpleApprox == 1
          Ylab{spmcls(ci)} = simple_approx( Yi , 1); 
        else
          Ylab{spmcls(ci)} = cat_vol_approx( Yi , 'nn' , vx_vol, 4); 
        end

      
        % smoother close to large GM structures
        Yws = cat_vol_smooth3X(Ylab{spmcls(ci)},16);
        Ylab{spmcls(ci)}  = Ylab{spmcls(ci)} .* (1-Ymx) + Yws .* (Ymx);

        % avoid crossing of tissue intensities
        if cio>0
          Ylab{spmcls(ci)} = min(Ylab{spmcls(ci)}, 0.95 * Ylab{spmcls(cio)} );
        end
        cio = spmcls(ci);
      end
      Ylab{6} = min(0,median(Ylab{3}(:)) - 2 * std(Ylab{3}(:))); % in Ym it is 0!
      
  
      
      % local intensity adaptation/correction
      if job.normInt
        % default values for normalized tissues
        T3thn = [1 2 3] / 3; 
      else
        % otherwise we use the average normal intensity 
        T3thn = [cat_stat_nanmedian( Ysrc( Yp0toC(Yp0c(:),1)>.9 & Yp0toC(Yp0(:),1)>.9 )) ...
                 cat_stat_nanmedian( Ysrc( Yp0toC(Yp0c(:),2)>.9 & Yp0toC(Yp0(:),2)>.9 )) ...
                 cat_stat_nanmedian( Ysrc( Yp0toC(Yp0c(:),3)>.9 & Yp0toC(Yp0(:),3)>.9 ))];
        
      end
      
      
      % Step-wise intensity normalization starting with 4 steps 
      % (WM+, WM-GM, GM-CSF, CSF-BG) with the folling basic structure:
      %    Yml = Yml + new_intensity_range_mask .* scaled_intenity_range_values
      % Although this kind of scaling is not optimal for the histogram it
      % allows much simpler use e.g. to estimate the differences to a (label) 
      % segmentation.
      Yml = zeros(size(Ysrc)); 
      Yml = Yml + ( (Ysrc>=Ylab{2}                ) .* ...
            (T3thn(3) + (Ysrc-Ylab{2}) ./ max(eps,Ylab{2}-Ylab{3}) .* max(eps,T3thn(3)-T3thn(2)) ));
      Yml = Yml + ( (Ysrc>=Ylab{1} & Ysrc<Ylab{2} ) .* ...
            (T3thn(2) + (Ysrc-Ylab{1}) ./ max(eps,Ylab{2}-Ylab{1}) .* max(eps,T3thn(3)-T3thn(2)) ));
      Yml = Yml + ( (Ysrc>=Ylab{3} & Ysrc<Ylab{1} ) .* ...
            (T3thn(1) + (Ysrc-Ylab{3}) ./ max(eps,Ylab{1}-Ylab{3}) .* max(eps,T3thn(2)-T3thn(1)) ));
      Yml = Yml + ( (Ysrc< Ylab{3}                ) .* ...
            (           (Ysrc-Ylab{6}) ./ max(eps,Ylab{3}-Ylab{6}) .* max(eps,T3thn(1)) ));
      Yml(isnan(Yml) | Yml<0)=0; Yml(Yml>10)=10;
      Yml = Yml / max(T3thn);
    end
  
  
    % #### denoising in intensity normalized data! ####
  
  
    
  
    %% write output
    %  ----------------------------------------------------------------------
    %  - to save some disk space we use int16 (range: -32768:1:32767)
    %  - although we scale images with lowest values in average to zero, 
    %    negative values are still possible and uint16 would just cut them
    %  - to have some simple image intensity range we set the WM to 100 and
    %    use 2 digits (pinfo 1e-2) to have some more details resulting in 
    %    a data range of -327.68:0.01:327.67 
    %  - uint16 would give no advantage here and we could also easily use 
    %    int8 (-128:1:127) or uint8(0:1:255) where this this time the uint8
    %    is more save in case of non-T1 modalities
    %  - the biasfield is only relevant for tests and to see if we get some
    %    smooth field without structural aspects, ie. if you see the head or 
    %    brain structures then the correction was not good
    stime = cat_io_cmd('  Write output','g5','',job.verb,stime); 
    switch job.dt, case 512, pinfo=1e-2; case 2, pinfo=1; case 16, pinfo=1; end 
    Vm  = Vo;  Vm.fname = Po2;  Vm.dt(1) = job.dt;  Vm.pinfo(1) = pinfo; 
    
    % original map but intensity normalized for fast comparison
    % - in case of T2w we need to integrate the ultra high CSF values 
    if job.write.org
      Vm.fname = Po2;  
      Yo2 = (Yo - Tmin) ./ (cat_stat_nanmedian(Yo(Yp0c>2.9 & Yp0>2.9)) * (1 + (modality==2)) - Tmin);
      spm_write_vol(Vm, max(-1,min(4,Yo2)) * 100);
      clear Yo2; 
    end
    
    % normalized images
    if job.write.GAS == 1  ||  job.write.GAS == 3
      Vm.fname = Pm;  
      spm_write_vol(Vm, max(-1,min(4,Ym0)) * 100);
    elseif job.write.GAS == 2  ||  job.write.GAS == 3
      Vm.fname = Pm;    
      spm_write_vol(Vm, max(-1,min(4,Ym)) * 100);
    end
    
    % local adaptive segmentation 
    % #### better limites for MT! and type of normalization  ####   
    if job.write.LAS
      Vm.fname = Pml;   
      spm_write_vol(Vm, max(0,min(1.25 + 2*(modality>1),Yml)) * 100);
    end
    if 1
      %% write simple segmentation with PVE class
      Yb  = cat_vol_smooth3X( Yp0>.5 , 4 )>.5; 
      if modality==1 
        Ymx = Yml .* Yb; % skull-stripping
        Ymx = max(Yb, min(3,round(Ymx * 6 / (1 + 2*(modality>1)) ) / 2)); 
        Ymx = min( Yb + 2*(Yp0>1.5),Ymx); 
        Ymx(Ymx==0.5) = 1;
      else
        % here we only create a simple 3 class label map
        [~,clso] = sort(T3th);
        Ymx = Yb .* (single(round(Yml/3*clso(3)*3)==clso(1))*1 ...
                   + single(round(Yml/3*clso(3)*3)==clso(2))*2 ...
                   + single(round(Yml/3*clso(3)*3)==clso(3))*3);
      end
      Vms = Vo;  Vms.fname = Po2;  Vms.dt(1) = 2;  Vms.pinfo(1) = 1e-1; Vms.fname = Pms;   
      spm_write_vol(Vms, Ymx);
    end
    
    % bias field map
    if job.write.biasfield
      Vm.fname = Pbf;   
      if job.write.GAS == 1  ||  job.write.GAS == 3   
        Ywx = Yw0; 
      elseif job.write.GAS == 2  ||  job.write.GAS == 3
        Ywx = Yw; 
      end
      % add scaling bars
      Ywx(:,:,1) = 1/1.5; Ywx(:,:,end) = 1.5; 
      spm_write_vol(Vm, max(0,min(4,Ywx/T3th(3))) * 100);
    end
  
    % #### LAS correction map ####
    if job.write.biasfield & 0
      Vm.fname = Plf;  
      Ywx = ( Ylab{1} + Ylab{2} + Ylab{3} ) ; 
      Ywx(:,:,1) = 1/1.1; Ywx(:,:,end) = 1.1; 
      spm_write_vol(Vm, max(0,min(4,Ywx)) * 100);
    end
  
    stime = cat_io_cmd('  ','g5','',job.verb,stime); 
    if job.verb, fprintf('%5.0fs\n',etime(clock,stime2)); end
  end
  fprintf('done.\n')
end

function Ya = simple_approx(Y,s,s2,red,Ymsk)
%simple_approx. Simple approximation by the closest Euclidean value.
%
%  [Yo,D,I] = simple_approx(Y[,s,Ymsk])
%  Y    .. input  image those zeros will be approximated by closes values
%  Ya   .. output image with appoximated values 
%  s    .. smoothing filter size (default = 1)
%  s2   .. smoothing filter size for distant values (defaut = 10)
%  Ymsk .. limit approximation (default = full image)
%

  if ~exist('s', 'var'), s = 1; end
  if ~exist('s2', 'var'), s2 = 10; end
  if ~exist('Ymsk', 'var'), Ymsk = true(size(Y)); end
  if ~exist('red', 'var'), red = 2; end
  
  % use lower resolution for faster processing 
  [Y,Ymsk,res] = cat_vol_resize( {Y,single(Ymsk)} , 'reduceV',1,red,16,'meanm');
  Ymsk = Ymsk > 0.5;
  s = s/red;

  Y = cat_vol_localstat(Y,Y~=0,1,1,4);

  % estimate closest object point
  [D,I] = cat_vbdist(single(Y~=0),Ymsk > 0); 
  D = max(0,min(1,(D - s2/2) / s2)); 
  
  % align (masked) non-object voxels with closest object value
  Ya = Y(I); 

  % smooth the result - correct for average to avoid smoothing boundary issues
  mnYo = median(Ya(Ya(:)~=0)); Ya = Ya - mnYo; Ya(~Ymsk) = 0; 
  Yas  = Ya; spm_smooth(Yas  , Yas  , repmat(s     ,1,3)); 
  Yas2 = Ya; spm_smooth(Yas2 , Yas2 , repmat(s * 10,1,3)); 
  Ya   = Yas .* (1-D) + Yas2 .* D;

  Ya = Ya + mnYo; Ya(~Ymsk) = 0; 

  Ya = cat_vol_resize( Ya , 'dereduceV', res);
  if red>1
    spm_smooth(Ya, Ya, repmat(red/2,1,3));
  end
end
