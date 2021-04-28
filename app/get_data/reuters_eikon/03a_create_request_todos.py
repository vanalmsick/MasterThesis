import os, sys, datetime
import pandas as pd


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################


def create_todo_list():
    aws_param = my.get_credentials(credential='aws')

    with my.postgresql_connect(aws_param) as conn:
        df_rics = my.sql_query(sql="SELECT * FROM reuters.ric_list", conn=conn)
    instrument_list = df_rics['ric'].tolist()
    n = 2
    instrument_list_of_list = [instrument_list[i:i + n] for i in range(0, len(instrument_list), n)]

    fields = ['TR.F.OthAssetsTot', 'TR.F.PbleAccrExpn', 'TR.F.AccrExpn', 'TR.F.AssetsHeldForSaleDiscOpsLTST', 'TR.F.LiabHeldForSaleDiscOpsLTST', 'TR.F.DivPble', 'TR.F.DerivFinInstrHedgeTot', 'TR.F.DerivLiabHedge', 'TR.F.TradeAcctPbleTot', 'TR.F.InvstTot', 'TR.F.IncTaxPbleLTST', 'TR.F.LoansRcvblTot', 'TR.F.FeesOth', 'TR.F.CashReceiptsPaymtNetCF', 'TR.F.DebtIssuedSTCF', 'TR.F.OpLeasePaymtDueInYr9', 'TR.F.PrefShrOutsIssue', 'TR.F.NonRecurPortOfEqEarn', 'TR.F.DPSComNetIssueCPO', 'TR.F.StockOthEqIssuedSoldCF', 'TR.F.OthAssetsLiabNetCF', 'TR.F.IncTaxFornCurr', 'TR.F.CompForDisasterPretax', 'TR.F.DiscOpsTaxImpacts', 'TR.F.NetIncDilInclExordItemsComTot', 'TR.F.UntaxedSpecialRsrv', 'TR.F.NetCashEndBal', 'TR.F.IncTaxOthKFASNLSTCurr', 'TR.F.ExordItems', 'TR.F.EBITDA', 'TR.F.OpLeasePaymtDueIn45Yr', 'TR.F.DebtIssuedReducedSTTotCF', 'TR.F.ComShrOutsIssue', 'TR.F.DiscOpsBefTaxTot', 'TR.F.EPSDilExclExordItemsComIssue', 'TR.F.StockComPrefOthRepurchRetiredCF', 'TR.F.DebtLTMatYr8', 'TR.F.CapLeaseMatIntrCosts', 'TR.F.PrefStockConvertNonRedeem', 'TR.F.DebtLTMat45Yr', 'TR.F.NetCashFlowDiscOpsOp', 'TR.F.MinIntrNonEq', 'TR.F.AmortIntangDefChrgCF', 'TR.F.DeprDeplAmortTot', 'TR.F.IntangGrossTot', 'TR.F.SGAUnclassif', 'TR.F.NonRecurIncExpnTot', 'TR.F.RentalExpn', 'TR.F.DivComCashPaid', 'TR.F.RepNonRecurItemNonGAAP', 'TR.F.RestrChrg', 'TR.F.BizRelFinRevOth', 'TR.F.Lvl1LiabFV', 'TR.F.EPSDilDiscOpsExordItems', 'TR.F.EstTaxImpactOnPretaxDiscOps', 'TR.F.DefChrg', 'TR.F.EmpFTEEquivPrdEnd', 'TR.F.OpLeasePaymtRemainMat', 'TR.F.DebtLTMatYr6Beyond', 'TR.F.ExordActivAfterTaxGL', 'TR.F.OpLeasePaymtDueInYr10', 'TR.F.NetIncAfterTax', 'TR.F.EPSBasicNonGAAPIssue', 'TR.F.PrepaidExpnCF', 'TR.F.SGAOthTot', 'TR.F.RentalOpLeaseExpn', 'TR.F.StockBasedCompExpnNetOfTaxSuppl', 'TR.F.AssetsSaleGLCF', 'TR.F.IncTaxOthByType', 'TR.F.IntangNetCF', 'TR.F.ComShrCapInclShrPremTot', 'TR.F.ComprEPSBasicIssue', 'TR.F.ComprEPSBasicIssueDR', 'TR.F.EPSBasicInclExordItemsComIssueCPO', 'TR.F.DerivFinInstrHedgeSoldPurchTotCF', 'TR.F.PrefShrTrezTotCurr', 'TR.F.IncTaxPaidReimbDirCF', 'TR.F.ComEqContrib', 'TR.F.EPSBasicDiscOpsExordItems', 'TR.F.DebtTot', 'TR.F.DebtLTMatYr5', 'TR.F.OthComprIncDiscOps', 'TR.F.StockPrefIssuedSoldCF', 'TR.F.ExordItemsAfterTaxCF', 'TR.F.EPSDilExclExordItemsComTot', 'TR.F.STDebtNotesPble', 'TR.F.ComprIncUnearnComp', 'TR.F.StockComPrefOthNetCF', 'TR.F.PPESoldCF', 'TR.F.BizSoldCF', 'TR.F.TotShHoldEq', 'TR.F.FXEffectsCF', 'TR.F.InvstSec', 'TR.F.StockComIssuedSoldCF', 'TR.F.NetCashFlowOp', 'TR.F.AfterTaxAdjSpecialRsrv', 'TR.F.EPSBasicNonGAAPIssueDR', 'TR.F.RetainedEarnTot', 'TR.F.CashPaymtBizActivCF', 'TR.F.ComprIncDivOnPrefStock', 'TR.F.DefTaxLiabLT', 'TR.F.AssetAllocFactorIssue', 'TR.F.OthComprIncPensRel', 'TR.F.AfterTaxAdjOthTot', 'TR.F.CapLeaseMatTot', 'TR.F.DeprInvstPropSuppl', 'TR.F.OpLeasePaymtDueInYr7', 'TR.F.DebtIssuedLTCF', 'TR.F.ComShrOutsIssueCurr', 'TR.F.MinIntrEq', 'TR.F.StockIssuanceRetNetExclOptWarrCF', 'TR.F.LimitedPartner', 'TR.F.NetCashFlowDiscOpsFin', 'TR.F.NetCashFlowFin', 'TR.F.IncTaxDom', 'TR.F.IntrPaidCashCashFlowDir', 'TR.F.ComShrOutsIssueCPO', 'TR.F.OthComprIncFornCcy', 'TR.F.IncTax', 'TR.F.IntangSoldCF', 'TR.F.EPSBasicExclExordItemsNormIssueDR', 'TR.F.TotCurrLiab', 'TR.F.CapLeaseMatDueIn1Yr', 'TR.F.IntangAccumAmortTot', 'TR.F.EPSBasicInclExordItemsComIssue', 'TR.F.AcctChgCF', 'TR.F.GenPartner', 'TR.F.InvstSecSoldMaturedCF', 'TR.F.DebtIssuedReducedLTSTCF', 'TR.F.FVAdjOthAssets', 'TR.F.IncAvailToComShr', 'TR.F.CapLeaseMatDueIn23Yr', 'TR.F.OpExpn', 'TR.F.OthLiabTotCF', 'TR.F.CapLeaseMatRemainMat', 'TR.F.NonGAAPAdjEBITDA', 'TR.F.HybridFinInstrEqPort', 'TR.F.PrefShrAuthIssue', 'TR.F.EPSBasicInclExordItemsComIssueDR', 'TR.F.AcctPbleAccrExpnCF', 'TR.F.InvstPropFV', 'TR.F.DivOnHybridFinInstrLiabPort', 'TR.F.InvstOthTot', 'TR.F.TfrToUntaxedRsrv', 'TR.F.ImpactOnTaxOfNormItems', 'TR.F.NormPretaxProf', 'TR.F.EPSBasicExclExordItemsComTot', 'TR.F.IslamicDebtLTST', 'TR.F.ProvTot', 'TR.F.ShHoldEqCom', 'TR.F.DefRevInc', 'TR.F.IntrDivRecdTotCF', 'TR.F.DebtReducedSTCF', 'TR.F.NonRecurItemsImpactOnIncTax', 'TR.F.DefTaxAssetLTST', 'TR.F.ShrUsedToCalcBasicEPSIssue', 'TR.F.PrefStockRedeemConvert', 'TR.F.NetCashContOps', 'TR.F.ComShrTrezIssue', 'TR.F.NonGAAPIncOp', 'TR.F.EarnParticipFactorIssue', 'TR.F.DPSComGrossIssueCPO', 'TR.F.OthOpIncTot', 'TR.F.PrepaidExpnTot', 'TR.F.OpLeasePaymtDueInYr3', 'TR.F.NetCashFlowDiscOpsTot', 'TR.F.Lvl3LiabFV', 'TR.F.InvstPermanent', 'TR.F.ComStockIssuedPaid', 'TR.F.ComStockTrezRepurch', 'TR.F.ProvDoubtAcctTot', 'TR.F.ComShrOutsIssueDR', 'TR.F.CapLeaseMatDueInYr6Beyond', 'TR.F.TotLiab', 'TR.F.DefIncTaxIncTaxCreditsCF', 'TR.F.CurrPortOfLTDebtCapLeases', 'TR.F.StockTotIssuanceRetNetCF', 'TR.F.SaleOfDiscOpsBefTaxGL', 'TR.F.MinIntr', 'TR.F.ShHoldEqParentShHoldTot', 'TR.F.AdExpn', 'TR.F.AllocDilNetIncInclExordItemsComIssue', 'TR.F.EarnAdjToNetIncOthExpnInc', 'TR.F.CashPaymtOthCF', 'TR.F.OthOpExpn', 'TR.F.RevalRsrv', 'TR.F.WarrConvertedCF', 'TR.F.EPSBasicNonGAAPTot', 'TR.F.Lvl2AssetsFV', 'TR.F.ImpairGoodw', 'TR.F.IntangExclGoodwGross', 'TR.F.RepNetIncNonGAAP', 'TR.F.EPSDilInclExordItemsComIssueCPO', 'TR.F.EPSBasicExclExordItemsNormIssueCPO', 'TR.F.NonGAAPEPSBasic', 'TR.F.DPSSpecialNetIssueDR', 'TR.F.TotLTCap', 'TR.F.EPSBasicExclExordItemsComIssue', 'TR.F.BVExclOthEq', 'TR.F.StockComRepurchRetiredCF', 'TR.F.ShrUsedToCalcEPSDilIssue', 'TR.F.Lvl1AssetsFV', 'TR.F.NetIncAfterMinIntr', 'TR.F.AmortOfCmptrSWSuppl', 'TR.F.EqIncLossInNetEarnCF', 'TR.F.DebtLTMatYr10', 'TR.F.InvstSecAFSHTMHFTTot', 'TR.F.LitigExpnSettle', 'TR.F.SGATot', 'TR.F.FVAdjInvstProp', 'TR.F.NonGAAPEPSDil', 'TR.F.LaborRelExpnInclStockBasedCompInSGA', 'TR.F.OpProv', 'TR.F.DebtLTMatYr7', 'TR.F.StockPrefRepurchRetiredCF', 'TR.F.EBITDANorm', 'TR.F.OthNonOpIncExpnTot', 'TR.F.ComEqTot', 'TR.F.InvstSecPurchCF', 'TR.F.NetCashFlowDiscOpsInvst', 'TR.F.Zakat', 'TR.F.IncBefTax', 'TR.F.TangTotEq', 'TR.F.PrefShrIssuedIssueCurr', 'TR.F.IncAvailToComExclExordItems', 'TR.F.GoodwCumlWrittOff', 'TR.F.EqOth', 'TR.F.IntangExclGoodwNetTot', 'TR.F.OpLeaseLiabCF', 'TR.F.StockBasedCompTaxBenefSuppl', 'TR.F.EPSDilNonGAAPTot', 'TR.F.HedgeRsrv', 'TR.F.EPSDilExclExordItemsNormTot', 'TR.F.DeprExpnTotSuppl', 'TR.F.OthComprIncOth', 'TR.F.AssetParticipFactorIssue', 'TR.F.OthComprIncStartingLine', 'TR.F.EPSDilInclExordItemsComIssueDR', 'TR.F.ComShrTrezIssueDR', 'TR.F.EqEarnLossBefTaxInclNonRecur', 'TR.F.ProfLossStartingLineCF', 'TR.F.NetCashFlowInvst', 'TR.F.InvstUnrealGL', 'TR.F.ImpairIntangExclGoodw', 'TR.F.ComprIncParentTot', 'TR.F.DPSComNetIssueDR', 'TR.F.IntrExpnHybridDebtInstrEq', 'TR.F.HybridFinInstrLiabCashFlow', 'TR.F.UnearnRevTot', 'TR.F.ESOPGuarPrefDefComp', 'TR.F.TotAssets', 'TR.F.ImpairFinInvst', 'TR.F.SGAExclRnD', 'TR.F.OpLeasePaymtDueInYr4', 'TR.F.FinAssetsUnrealGLCF', 'TR.F.DebtIssuedLTSTCF', 'TR.F.PrefStockNonRedeem', 'TR.F.TaxPbleCF', 'TR.F.ComprEPSBasicIssueCPO', 'TR.F.DistribForGenPartners', 'TR.F.CapLeaseMatDueInYr5', 'TR.F.DPSComGrossIssueDR', 'TR.F.CapLeaseMatDueInYr9', 'TR.F.IntangExclGoodwAccumAmortTot', 'TR.F.EPSDilNonGAAPIssueDR', 'TR.F.PrefShrIssuedTotCurr', 'TR.F.StockOthEqNetCF', 'TR.F.OpLeasePaymtDueInYr6Beyond', 'TR.F.StockOthEqRepurchRetiredCF', 'TR.F.PrefShrOutsIssueCurr', 'TR.F.ShrBasedPaymtCF', 'TR.F.ImpairPPEInclIntangCF', 'TR.F.OthNonCashItemsReconcAdjCF', 'TR.F.OpLeasePaymtIntrCostImputedIntr', 'TR.F.CashReceiptsBizActivCF', 'TR.F.SaleOfDiscOpsNetGL', 'TR.F.OpLeasePaymtDueInYr8', 'TR.F.EBITNorm', 'TR.F.OpLeasePaymtDueIn23Yr', 'TR.F.DeprDeplAmortInclImpairCF', 'TR.F.CapLeaseMatDueInYr10', 'TR.F.OpLeasePaymtDueInYr2', 'TR.F.AmortOfBrandsPatentsSuppl', 'TR.F.EPSDilNonGAAPIssueCPO', 'TR.F.IncTaxOthByRegion', 'TR.F.IntangAmortOthSuppl', 'TR.F.InvstInAssocJVsUnconsolSubs', 'TR.F.PensBenefOverfundedTot', 'TR.F.EBITDAOpLeasePaymt', 'TR.F.OpLeasePaymtDueInYr6', 'TR.F.IncTaxDomCurr', 'TR.F.IncBefDiscOpsExordItems', 'TR.F.EPSBasicExclExordItemsNormTot', 'TR.F.ImpairInvstProp', 'TR.F.AuditRelFees', 'TR.F.DPSSpecialGrossIssueCPO', 'TR.F.IncTaxFornDef', 'TR.F.ImpairFixedAssets', 'TR.F.OthAssetsCF', 'TR.F.DPSComNetIssue', 'TR.F.PropOthTax', 'TR.F.DebtLTMatIn1Yr', 'TR.F.DPSSpecialGrossIssueDR', 'TR.F.ComprIncPensLiab', 'TR.F.AmortOfLicensesFranchisesSuppl', 'TR.F.ShrUsedToCalcBasicEPSIssueCPO', 'TR.F.ComShrIssuedIssueCurr', 'TR.F.ShrUsedToCalcDilEPSTot', 'TR.F.DiscOpsNetIncExpn', 'TR.F.OthThanTempImpairLossesOnInvst', 'TR.F.ComShrTrezIssueCPO', 'TR.F.IntangNetTotCF', 'TR.F.NormNetIncContOps', 'TR.F.TangBV', 'TR.F.MinIntrCF', 'TR.F.OthLiab', 'TR.F.CAPEXTot', 'TR.F.IncTaxOthDef', 'TR.F.STDebtCurrPortOfLTDebt', 'TR.F.StockComNetCF', 'TR.F.ComprIncOthTot', 'TR.F.DebtInclPrefEqMinIntrTot', 'TR.F.DPSSpecialIssue', 'TR.F.OthComprIncAssocCo', 'TR.F.ShrUsedToCalcBasicEPSIssueDR', 'TR.F.ComShrIssuedIssue', 'TR.F.EBIT', 'TR.F.TotCurrAssets', 'TR.F.ImpactsChgInAcctStd', 'TR.F.Lvl2LiabFV', 'TR.F.ImpairDefCosts', 'TR.F.OthComprIncUnrealInvstGL', 'TR.F.DebtRestrGL', 'TR.F.NegConsolWrittOffAgainstRsrv', 'TR.F.OpLeaseLiabLTST', 'TR.F.CapLeaseCurrPort', 'TR.F.TotLiabEq', 'TR.F.OthOpExpnIncNet', 'TR.F.DeprTot', 'TR.F.CapLeaseObligLT', 'TR.F.DilIncAvailToComExclExordItems', 'TR.F.DebtLTMatRemain', 'TR.F.EPSDilInclExordItemsComTot', 'TR.F.NetCashFlowDiscOpsOth', 'TR.F.FXGL', 'TR.F.EmpPartTimePrdEnd', 'TR.F.PrefShrTrezIssueCurr', 'TR.F.CostsAssocWithIPOsMergers', 'TR.F.PrefStockRedeemTot', 'TR.F.OpLeasePaymtDueInYr1', 'TR.F.NetIncBefMinIntr', 'TR.F.ProvOthThanPensPostRet', 'TR.F.DefShr', 'TR.F.GoodwCostInExcessOfAssetsPurchNet', 'TR.F.StockBasedCompExpnPretaxSuppl', 'TR.F.AmortOfCapRnDSuppl', 'TR.F.AmortTotSuppl', 'TR.F.EPSDilExclExordItemsComIssueDR', 'TR.F.AmortOfDefChrgTot', 'TR.F.RecognitionOfNegGoodw', 'TR.F.StockPrefNetCF', 'TR.F.SaleAcqOfGroupCoGL', 'TR.F.PrefStockIssuedForESOP', 'TR.F.SWDevCostsCF', 'TR.F.TotRevBizActiv', 'TR.F.EarnAllocFactorBasicIssue', 'TR.F.AcqDispOfBizAssetsSoldAcqNetCF', 'TR.F.LaborRelExpnSuppl', 'TR.F.DebtIssuedReducedLTCF', 'TR.F.PPENetTot', 'TR.F.NonRecurUnusualItemsTaxImpact', 'TR.F.CashSTInvstTot', 'TR.F.DilAdj', 'TR.F.DebtLTMat23Yr', 'TR.F.FornCcyTranslAdjAccum', 'TR.F.ComprEPSDilIssue', 'TR.F.ShrUsedToCalcDilEPSIssueCPO', 'TR.F.DefTaxInvstTaxCreditsLT', 'TR.F.OthComprIncRevalOfTangIntang', 'TR.F.OpProfBefNonRecurIncExpn', 'TR.F.EPSDilInclExordItemsComIssue', 'TR.F.OthComprIncHedgeGL', 'TR.F.IncTaxDomDef', 'TR.F.DebtLTSTIssuanceRetTotCF', 'TR.F.AmortOfIntangInclGoodwTot', 'TR.F.GoodwAmortSuppl', 'TR.F.EPSBasicNonGAAPIssueCPO', 'TR.F.ComprIncBefMinIntrTot', 'TR.F.InvstAssocCoJVsCF', 'TR.F.PriorityDivIssue', 'TR.F.IncTaxForn', 'TR.F.OthComprIncNetOfTaxTot', 'TR.F.GoodwAccumAmort', 'TR.F.DPSSpecialNetIssue', 'TR.F.EmpAvg', 'TR.F.ComShrTrezTot', 'TR.F.DebtLTMatYr6', 'TR.F.DebtReducedLTSTCF', 'TR.F.MinIntrJVsNetCF', 'TR.F.NonRecurAdjOp', 'TR.F.IncTaxPaidReimbIndirCF', 'TR.F.DebtReducedLTCF', 'TR.F.RepTaxImpactNonRecurItemsNonGAAP', 'TR.F.DistribForPrefShr', 'TR.F.EqInEarnLossOfAffilAfterTax', 'TR.F.NonRecurIncExpnOthTot', 'TR.F.OpLeasePaymtTot', 'TR.F.AmortOfFinLeaseROUAssetsSuppl', 'TR.F.DebtLTMatYr9', 'TR.F.EPSDilNonGAAPIssue', 'TR.F.ProvForIncTaxByRegionTot', 'TR.F.AcqOfBizCF', 'TR.F.RestrAcctCash', 'TR.F.CapLeaseMatDueInYr7', 'TR.F.USGAAPAdj', 'TR.F.DiscOpsGLNetOfTaxCF', 'TR.F.PrefStockTrezRepurch', 'TR.F.InvstSecSoldPurchNetTotCF', 'TR.F.EPSDilExclExordItemsComIssueCPO', 'TR.F.NonRecurPortOfEqEarnBefTax', 'TR.F.NormAfterTaxProf', 'TR.F.AmortOfIntangInSGA', 'TR.F.IntangPurchAcqCF', 'TR.F.EPSBasicInclExordItemsComTot', 'TR.F.AmortOfDefCostsSuppl', 'TR.F.IncTaxExpnCF', 'TR.F.CapLeaseMatExecutoryCosts', 'TR.F.IntangTotNet', 'TR.F.EPSDilExclExordItemsNormIssue', 'TR.F.CapLeaseMatDueInYr4', 'TR.F.CashFlowOpBefChgInWkgCap', 'TR.F.AssetAccruals', 'TR.F.TaxFees', 'TR.F.NonClassifCashFlows', 'TR.F.OthInvstCashFlow', 'TR.F.EPSBasicExclExordItemsComIssueCPO', 'TR.F.NonCashItemsReconcAdjCF', 'TR.F.ImpairTangIntangFixedAssets', 'TR.F.EPSDilExclExordItemsNormIssueCPO', 'TR.F.PostEmpBenefPensOth', 'TR.F.DeprAmortSuppl', 'TR.F.CAPEXNetCF', 'TR.F.CapLeaseMatDueInYr3', 'TR.F.ComprIncAccumTot', 'TR.F.ComprEPSDilIssueCPO', 'TR.F.DeprInSGA', 'TR.F.ComShrOutsTot', 'TR.F.PrefShrOutsTotCurr', 'TR.F.DivPaidCashTotCF', 'TR.F.NetCashBegBal', 'TR.F.DivPrefCashPaid', 'TR.F.ComShrTrezIssueCurr', 'TR.F.DefTaxLiabInUntaxedRsrv', 'TR.F.OthRsrvEqTot', 'TR.F.NetBookCap', 'TR.F.IntrPaidCash', 'TR.F.CapLeaseMatDueInYr6', 'TR.F.AmortOfIntangExclGoodwTot', 'TR.F.LaborRelExpnTot', 'TR.F.IncTaxRcvbl', 'TR.F.InvstProp', 'TR.F.OthFinCashFlow', 'TR.F.PPEPurchCF', 'TR.F.PrefShrTrezIssue', 'TR.F.StockComPrefOthIssuedSoldCF', 'TR.F.IncTaxForTheYrCurr', 'TR.F.InvstExclLoansCF', 'TR.F.NormNetIncBottomLine', 'TR.F.ComprEPSDilIssueDR', 'TR.F.AuditorFees', 'TR.F.ComStockShrPremInclOptionRsrv', 'TR.F.CashReceiptsPaymtAssocCF', 'TR.F.IntrExpn', 'TR.F.ComShrAuthIssue', 'TR.F.PrefShrIssuedIssue', 'TR.F.PrefShHoldEq', 'TR.F.SaleOfTangIntangFixedAssetsGL', 'TR.F.CapLeaseMatDueInYr2', 'TR.F.OptExercisedCF', 'TR.F.DPSSpecialNetIssueCPO', 'TR.F.CurrPortOfLTDebtExclCapLease', 'TR.F.ShrUsedToCalcBasicEPSTot', 'TR.F.ComShrIssuedTot', 'TR.F.OthComprIncUnearnInc', 'TR.F.EarlyExtingOfLeaseRelDebtsGL', 'TR.F.ComShrIssuedIssueDR', 'TR.F.PPENetCF', 'TR.F.TotBookCap', 'TR.F.ComprIncAttribToMinIntrTot', 'TR.F.ComStockHeldInESOTESOPDefComp', 'TR.F.EPSDilExclExordItemsNormIssueDR', 'TR.F.PrefStockRedeemTempEq', 'TR.F.NetIncBasicInclExordItemsComTot', 'TR.F.FVAdjFinInvst', 'TR.F.NonGAAPAdjNetEarn', 'TR.F.CapLeaseMatDueInYr8', 'TR.F.RcvblOth', 'TR.F.EPSBasicExclExordItemsNormIssue', 'TR.F.ComShrIssuedIssueCPO', 'TR.F.EPSBasicExclExordItemsComIssueDR', 'TR.F.OthAssets', 'TR.F.DebtLTMatYr2', 'TR.F.DiscOpsBefTaxIncExpn', 'TR.F.ComShHoldNum', 'TR.F.NonGAAPRev', 'TR.F.OthComprIncIncTax', 'TR.F.ParticipWgtPrimaryShrForEPSCalcIssue', 'TR.F.Lvl3AssetsFV', 'TR.F.NonGAAPOpMargPct', 'TR.F.ComEqParentShHold', 'TR.F.AllocNetIncInclExordItemsComIssue', 'TR.F.RepNormNetInc', 'TR.F.OthLiabTot', 'TR.F.ExordItemsTaxImpact', 'TR.F.DPSComGrossIssue', 'TR.F.GoodwCostInExcessOfAssetsPurchGross', 'TR.F.PrefEqContrib', 'TR.F.NonRecurAdjNonOp', 'TR.F.DiscOpsNetOfTaxTot', 'TR.F.IncTaxDef', 'TR.F.WkgCapCF', 'TR.F.NetChgInCashTot', 'TR.F.CapLeaseMatDueIn45Yr', 'TR.F.NetCashDiscOps', 'TR.F.ShrUsedToCalcDilEPSIssueDR', 'TR.F.AmortOfGoodwTot', 'TR.F.DebtLTMatYr4', 'TR.F.SupplAdjOp', 'TR.F.DebtLTMatYr3', 'TR.F.EqNonContribRsrvRetainedEarn', 'TR.F.DeprDeplPPECF', 'TR.F.AdExpnSuppl', 'TR.F.NetOpAssets', 'TR.F.OpLeasePaymtDueInYr5', 'TR.F.OpExpnTot', 'TR.F.TotCap', 'TR.F.Prov', 'TR.F.SupplAdjNonOp', 'TR.F.MinIntrTot', 'TR.F.CashCashEquivTot', 'TR.F.RevBizRelActivOthTot', 'TR.F.IntrDivRecdTotCFDir', 'TR.F.IntrCap', 'TR.F.FinIncExpnCF', 'TR.F.AcctNotesRcvblTradeGrossTot', 'TR.F.PPEInclIntangGLCF', 'TR.F.ConvertDebtLT', 'TR.F.CustLiabOnAcceptAssets', 'TR.F.PrefStockLiabPortLT', 'TR.F.CapAdeqCoreTier1Pct', 'TR.F.PPEAccumDeprTot', 'TR.F.OthMandatRedeemEqInstrLT', 'TR.F.CashCashEquiv', 'TR.F.LiqCovRatioBasel3Pct', 'TR.F.CapAdeqTotPct', 'TR.F.ContrLiabTot', 'TR.F.CapAdeqTier2Value', 'TR.F.FXGLNonBiz', 'TR.F.CustLiabOnAcceptLiab', 'TR.F.InvstSecGLCF', 'TR.F.DebtNonConvertLT', 'TR.F.ContrAssetsTot', 'TR.F.LTDebtExclCapLease', 'TR.F.HybridFinInstrLiabLT', 'TR.F.CapAdeqTier3Pct', 'TR.F.DebtLTMatTot', 'TR.F.CapAdeqTotValue', 'TR.F.PPEGrossTot', 'TR.F.AssetsUnderMgmtAUM', 'TR.F.EquipOccupExpnBankFinOth', 'TR.F.InvstPropExclCAPEXSoldPurchNetCF', 'TR.F.ImpairFinFixedAssetsOpCF', 'TR.F.CapAdeqTier1Value', 'TR.F.CapAdeqCoreTier1Value', 'TR.F.CapAdeqTier3Value', 'TR.F.InvstTradPropRealzGLCF', 'TR.F.CapAdeqTier1Pct', 'TR.F.CashPaymtEmpCF', 'TR.F.DebtLTTot', 'TR.F.NetStableFundingRatioBasel3Pct', 'TR.F.HybridFinInstrLiabCurrPort', 'TR.F.RiskWgtAssets', 'TR.F.CapAdeqTier2Pct', 'TR.F.InvstAssetsTot', 'TR.F.MandatRedeemTrustCertfLT', 'TR.F.CapAdeqHybridTier1', 'TR.F.LevRatioBasel3Pct', 'TR.F.WgtCostOfDebtPct', 'TR.F.CreditExposure', 'TR.F.IntrExpnNetOfCapIntr', 'TR.F.NetDebt', 'TR.F.InvstPropGross', 'TR.F.InvstPropAccumDepr', 'TR.F.ProvDoubtTradeAcctTradeNotesPbleTot', 'TR.F.IntrExpnOnCapFinGross', 'TR.F.CostOfOpRev', 'TR.F.IncTaxPaidReimbCFSuppl', 'TR.F.PPEExclROUTangCapLeaseGross', 'TR.F.InvntTot', 'TR.F.IntrEarnAssetsAvg', 'TR.F.LoansNonPerfOverdue', 'TR.F.COGSTot', 'TR.F.AccrExpnCF', 'TR.F.FeesCommExpn', 'TR.F.InvstSecSoldMaturedOpCF', 'TR.F.PPEExclROUTangCapLeaseAccumDepr', 'TR.F.LicensesFranchisesNet', 'TR.F.TotOpLeaseLiab', 'TR.F.SolvMargRatioInsur', 'TR.F.LoansNonPerfOverdue0To90Days', 'TR.F.ContrAssetsCF', 'TR.F.LoansPerfNonPerfNotImpair', 'TR.F.WkgCapCFDirSuppl', 'TR.F.InvstSecHFT', 'TR.F.SecPurchUnderRepo', 'TR.F.AcctPbleCF', 'TR.F.LoansPerfNotImpairNotPastDue', 'TR.F.CashDivPaidComStockBuybackNet', 'TR.F.CmptrSWIntangNet', 'TR.F.NonIntrFinIncExpnOthNet', 'TR.F.CmptrSWIntangGross', 'TR.F.SecHeldUnderCollat', 'TR.F.CollatFinAgrmtRepoFinLoansLiab', 'TR.F.DivProvPaidCom', 'TR.F.InvstSecOpCF', 'TR.F.LoansNonPerfImpair', 'TR.F.LoansImpairNonPerf', 'TR.F.IntrIncNonBank', 'TR.F.LoansImpairSubPerfPerfButImpair', 'TR.F.CollatAgrmtAssets', 'TR.F.ImpairAssetsImpairLoansOth', 'TR.F.AcctRcvblCF', 'TR.F.NonPerfAssetsLoansOth', 'TR.F.NonIntrFinIncExpnTot', 'TR.F.FinIncExpnNetTot', 'TR.F.BillingsInExcessOfCostsTot', 'TR.F.LoansNonPerfOverdueOver90Days', 'TR.F.TradAcct', 'TR.F.SolvRatioInsur', 'TR.F.InvstSecAFSHTM', 'TR.F.IntrDivRecdCFSuppl', 'TR.F.NonPerfAssetsOthThanLoans', 'TR.F.IntangOthNet', 'TR.F.COGSExclDepr', 'TR.F.COGSInclOpMaintUtilTot', 'TR.F.LicensesFranchisesAccumAmort', 'TR.F.FinOpLeaseLiabTot', 'TR.F.InvstPropUnrealGLCF', 'TR.F.InvstLoansBizGL', 'TR.F.OthPbleTot', 'TR.F.IslamicRcvblST', 'TR.F.DivInvstInc', 'TR.F.ContrLiabCF', 'TR.F.EarnAssets', 'TR.F.IslamicInvstDepos', 'TR.F.ROUIntangNet', 'TR.F.IntrPaidCFSuppl', 'TR.F.BrandsPatentsAccumAmort', 'TR.F.IslamicRcvblLT', 'TR.F.NonOpRentalInc', 'TR.F.TradLiab', 'TR.F.InvstSecDesigAtFVOth', 'TR.F.LeveredFOCF', 'TR.F.DeprDeplAmortCF', 'TR.F.IslamicInc', 'TR.F.ImpairAssetsExclImpairLoans', 'TR.F.CmptrSWIntangAccumAmort', 'TR.F.STBankBorrowExclCollatFin', 'TR.F.ProvImpairForLoanLosses', 'TR.F.CommFeesSecActiv', 'TR.F.SaleOfInvstHeldForSaleMatTradGL', 'TR.F.ImpairAssetsTot', 'TR.F.LicensesFranchisesGross', 'TR.F.InvstSecPurchOpCF', 'TR.F.ROUIntangAccumAmort', 'TR.F.BrandsPatentsNet', 'TR.F.PPEExclROUTangCapLeaseNet', 'TR.F.LoansImpairTot', 'TR.F.FOCF', 'TR.F.IntrExpnNetOfIntrInc', 'TR.F.DebtInclFinOpLeaseLiab', 'TR.F.FeesCommInc', 'TR.F.IntangOthGrossTot', 'TR.F.CostOfFinRelOp', 'TR.F.BrandsPatentsGross', 'TR.F.AmortOfDefFinChrg', 'TR.F.SaleOfLeasedFixedAssetsGL', 'TR.F.IncIslamicFinInvst', 'TR.F.GLOnSaleOfFinInstrOth', 'TR.F.SecBorrow', 'TR.F.SettleOfLoansGL', 'TR.F.CollatAgrmtAssetsCF', 'TR.F.CollatAgrmtRevRepoFinLiabCF', 'TR.F.COGSUnclassif', 'TR.F.IntangOthAccumAmort', 'TR.F.IntrBearLiabAvg', 'TR.F.RepoLiabCF', 'TR.F.InvstSecAFSTot', 'TR.F.SecSoldUnderRepoFedFundsPurch', 'TR.F.CashReceiptsOthCustCF', 'TR.F.CashReceiptsSalesOfGoodsSrvcCF', 'TR.F.CollatAgrmtRevRepoSBLiabCF', 'TR.F.COGSOthTot', 'TR.F.DerivHedgeBizFVAdjGL', 'TR.F.LoanLossProvInclImpairCF', 'TR.F.AmortInCOGS', 'TR.F.LaborRelExpnInclStockBasedCompInCOGS', 'TR.F.RsrvForLoanLosses', 'TR.F.ROUIntangGross', 'TR.F.FinLeaseRcvblNetLTST', 'TR.F.IntrPble', 'TR.F.LoansGross', 'TR.F.InvstSecBizGL', 'TR.F.DeposTot', 'TR.F.DerivHedgeBizRealzGL', 'TR.F.DeprInCOGS', 'TR.F.BankRelLoansNet', 'TR.F.TradeAcctTradeNotesRcvblTot', 'TR.BSPeriodEndDate', 'TR.Revenue.date']
    properties = {'Sdate': '0', 'Edate': '1995-01-01', 'FRQ': 'FQ', 'Period': 'FQ0', 'Curn': 'USD', 'Scale': 0}  # 'Period': 'FQ0' will ensure just the latest statement avalable in each period no pre-statements

    task_id = 1
    date_iso = datetime.datetime.now().isoformat()

    new_todos = []
    for instruments in instrument_list_of_list:
        new_todos.append([task_id, str(instruments), str(fields), str(properties), 0, date_iso])
    new_todos = pd.DataFrame(new_todos, columns=['task_id', 'req_instruments', 'req_fields', 'req_parameters', 'status', 'last_updated'])

    with my.postgresql_connect(aws_param) as conn:
        my.df_insert_sql(conn, df=new_todos, table='reuters.data_request_list')


if __name__ == '__main__':
    create_todo_list()